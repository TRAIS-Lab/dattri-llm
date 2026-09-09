"""DVEmb (Data Value Embedding) trajectory-aware attribution.

Unlike TracIn -- which simply dots the train gradient at the step a sample was
used against the test gradient at that *same* checkpoint -- DVEmb accounts for
how a training update at step ``t_s`` keeps propagating through every
*subsequent* training step before reaching the final model ``theta_T``.  Following
"Capturing the Temporal Dependence of Training Data Influence"
(https://arxiv.org/abs/2412.09538), the influence of a training sample ``z*``
used at step ``t_s`` on a test point ``z_val`` is

    I(z*, t_s) = eta_{t_s} * dl(theta_T, z_val)^T
                 [ prod_{k=t_s+1}^{T-1} (I - eta_k H_k) ] dl(theta_{t_s}, z*)        (1)

i.e. the train gradient at ``t_s`` is pushed forward through the product of the
SGD Jacobians ``(I - eta_k H_k)`` of every later step ``k`` and then dotted with
the test gradient taken at the **final** model ``theta_T`` (capital ``T`` =
``final_step``).  The per-step Hessian is the Gauss-Newton / empirical-Fisher
approximation built from that step's recorded per-sample gradients,

    H_k ~ (1/c) sum_{z in B_k} g_hat(theta_k, z) g_hat(theta_k, z)^T          (2)

where ``g_hat`` is the *recorded* per-sample gradient and ``c`` its per-sample loss
weight: the empirical Fisher must use the **true** per-sample gradients, so when
the gradients were recorded under a **mean** loss (``g_hat = dl / B``, ``c = 1/B``)
the sum of recorded outer products is rescaled by the step's batch size ``B``,
while under a **sum** loss (``g_hat = dl``, ``c = 1``) it is used as-is -- see the
``loss_reduction`` argument of :meth:`attribute`.

Setting every ``H_k = 0`` recovers TracIn (eta * <g_test, g_train>); the Fisher
factors are exactly the "training dynamics" correction DVEmb adds.

The ``hessian_mode`` argument selects the structure of (2): ``"full"``
(default) uses the exact rank-1 sum over *concatenated* per-sample gradients,
whose off-diagonal blocks couple layers through the training dynamics;
``"diagonal"`` zeroes the cross-layer blocks (block-diagonal per layer), which
is the approximation made by the official DVEmb implementation.

**Computation.** The bilinear form in (1) can be evaluated by carrying the
Fisher product on either side; the ``propagation`` argument selects which.
Both sweep the recorded training steps from latest to earliest and produce
**identical** scores.

With ``propagation="train"`` (the default) the product is applied to the
*train* side, yielding the paper's **data value embedding** per training sample,

    e_{t_s}(z*) = eta_{t_s} * [ prod_{k=t_s+1}^{T-1}(I - eta_k H_k) ]^T
                  g_hat(theta_{t_s}, z*),

so a row's score against any test column is simply ``<e, dl(theta_T, z_val)>``
-- an ordinary inner-product attribution.  This is how DVEmb plugs into
:class:`~dattri_llm.attribution.base.BaseInnerProductAttributor`: the sweep
(:meth:`DVEmbAttributor.embed_trajectory`) is a trajectory-level train-side
transform yielding embedding blocks, and scoring is the inherited loop with
the test side materialized.  The sweep carries the explicit accumulated
operator ``M_{t_s}`` with ``I - M_{t_s} = [ prod_{k>t_s}(I - eta_k H_k) ]^T`` --
a dense ``(d, d)`` matrix over the concatenated layer dimension, updated as
``M <- M + eta_{t_s} sum_{zinB_{t_s}} (g_hat(z) - M g_hat(z)) g_hat(z)^T``.  This
costs ``d^2`` memory (use projected gradients or ``layer_name`` to keep ``d``
small) but makes the embeddings test-independent and persistable
(:meth:`DVEmbAttributor.cache_dvemb`).

With ``propagation="test"`` the product is applied to the *test* side: for
every test column a running parameter-space vector

    w_{t_s} = [ prod_{k=t_s+1}^{T-1}(I - eta_k H_k) ]^T dl(theta_T, z_val)

is initialised at the final-model test gradient.  At each step ``t_s``
(descending) the rows for the train samples recorded there are
``eta_{t_s} * <g(z*), w_{t_s}>``, after which ``w`` is advanced by that step's
full Fisher factor.  This is matrix-free -- the product ``prod(I - eta H)`` is
never materialised -- but ties the sweep to the given test set.

This is the **basic** DVEmb estimator -- it materialises the per-layer gradients
and propagates the exact (Fisher-approximated) product.  Influence-checkpointing
and the low-rank embedding compression of the paper are deliberately omitted.

The result is an :class:`~dattri_llm.attribution.score.AttributionScore` whose
rows are ``(train_hash, step)`` pairs (one row per recorded checkpoint of a
sample, stamped with its step) and whose columns are the test-sample hashes in
store order -- identical bookkeeping to TracIn and the K-FAC family.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar

import torch
from tqdm.auto import tqdm

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.gradient.datasets import resolve_steps
from dattri_llm.gradient.gradient import Gradient
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.utils.cache import CACHE_RESIDENCIES

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import DiskGradientSource

LearningRate = float | Mapping[int, float]
StreamBlock = tuple[int, Gradient, list[str]]

_LR_SCHEDULE_FILE = "lr_schedule.json"


def _dense_float(block: Gradient) -> Gradient:
    """Materialize every layer of *block* to a float32 ``(B, d)`` tensor."""
    return block.materialize().map_layers(lambda _n, v, _t: v.float())


class DVEmbAttributor(BaseInnerProductAttributor):
    """DVEmb (Data Value Embedding) attributor.

    Scores every training record (at the step it was recorded) against every
    test record, correcting the raw TracIn inner product for how the update
    propagates through all *later* training steps up to the final model
    ``theta_T`` -- see the module docstring for the score definition.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory.
        task: Required by the on-the-fly :meth:`cache` / :meth:`attribute`
            (supplies the model, the loss, and the optional ``target_func`` for
            the test side); unused by :meth:`attribute_from_cache`.

    The learning-rate schedule ``eta`` is a per-attribution argument of
    :meth:`attribute` / :meth:`attribute_from_cache` (a float for a constant
    schedule or a ``{step: eta_step}`` mapping); it enters both the per-step score
    scale ``eta_{t_s}`` and the Fisher factors ``(I - eta_k H_k)``, so it must match
    the schedule the gradients were collected under.

    Layer selection happens at **capture** (the ``hook_config`` of the live
    methods, or whatever was hooked when the cache was collected).  By default
    every stored layer enters the Fisher and the score;
    :meth:`attribute_from_cache` additionally takes a ``layer_name`` read-time
    filter to score a subset of the stored layers.
    """

    algorithm: ClassVar[str] = "DVEmb"

    # ------------------------------------------------------------------ #
    # Collection                                                           #
    # ------------------------------------------------------------------ #

    def checkpoints(self) -> list[int]:
        """DVEmb regenerates the trajectory from the task's first checkpoint."""
        n_ckpt = self.num_checkpoints()
        if n_ckpt > 1:
            warnings.warn(
                f"DVEmb regenerates the trajectory live from checkpoint 0; the "
                f"other {n_ckpt - 1} provided checkpoint(s) are ignored.",
                stacklevel=2,
            )
        return [0]

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: str | None = None,
        hook_config: HookManagerConfig | None = None,
        offload_interval: int = 1,
    ) -> list[tuple[str, str]]:
        """Run the training trajectory live and cache the gradients DVEmb needs.

        DVEmb is trajectory-aware, so it cannot collect both sides at one
        checkpoint.  It needs (a) the per-step **train** gradients along the
        trajectory and (b) the **test** gradients at the *final* model
        ``theta_T``, which only exists once training finishes.  This method
        therefore drives one live training pass from the task's first
        checkpoint, offloading each step's train gradient, and then -- with the
        model now at ``theta_T`` -- a frozen pass over the test set.  The
        per-step learning rates actually applied are recorded beside the train
        store so :meth:`attribute_from_cache` can check the configured schedule.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            cache_dir: Parent directory for ``train_grads``/``test_grads``;
                defaults to ``args.output_dir``.
            hook_config: Capture configuration for both streamers.
            offload_interval: Steps accumulated per gradient file.  ``1``
                (default) writes one file per step -- best for DVEmb's per-step
                sweep (no redundant multi-step file reloads).

        Returns:
            ``[(train_gradients_dir, test_gradients_dir)]``.
        """
        self.require_task("cache")
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        train_dir = str(pathlib.Path(cache_dir) / "train_grads")
        test_dir = str(pathlib.Path(cache_dir) / "test_grads")
        recorded_lr = self.collect_trajectory(
            GradientStorageManager(train_dir),
            GradientStorageManager(test_dir),
            train_dataset,
            test_dataset,
            hook_config=hook_config,
            offload_interval=offload_interval,
        )
        self._write_lr_schedule(train_dir, recorded_lr)
        return [(train_dir, test_dir)]

    def collect_trajectory(
        self,
        train_store: GradientStorageManager,
        test_store: GradientStorageManager,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        offload_interval: int = 1,
    ) -> dict[int, float]:
        """Collect the training trajectory + final-model test gradients.

        Drives the live trajectory (``theta_0 -> theta_T``) into *train_store*
        and the frozen ``theta_T`` test pass into *test_store* (any residency),
        and returns the per-step learning rates actually applied -- the
        schedule the DVEmb Fisher product must match.
        """
        self.checkpoints()
        self.load_checkpoint(0)
        train_streamer = self.generate_train_rep(
            train_dataset,
            enable_update=True,
            hook_config=hook_config,
        )
        self.collect_gradients(
            train_streamer,
            train_store,
            offload_interval=offload_interval,
        )
        self.collect_gradients(
            self.generate_test_rep(test_dataset, hook_config=hook_config),
            test_store,
            offload_interval=offload_interval,
        )
        return train_streamer.learning_rates

    def cache_dvemb(
        self,
        train_gradients_dir: str,
        dvemb_dir: str | None = None,
        *,
        selected_training_steps: Iterable[int] | None = None,
        hessian_mode: str = "full",
        final_step: int | None = None,
        loss_reduction: str = "mean",
        verbose: bool = False,
        layer_name: str | list[str] | None = None,
        learning_rate: LearningRate = 1.0,
    ) -> str:
        """Turn a stored gradient trajectory into persisted **data value embeddings**.

        One train-side sweep (:meth:`embed_trajectory`) over an existing
        per-step gradient store -- written by :meth:`cache` or by any training
        run wrapped with hooks -- turning every train record into its embedding
        ``e = eta * prod_{k>t_s}(I - eta_k H_k)^T g_hat`` and storing it in
        ``dvemb_dir`` as materialized per-layer :class:`Gradient` records with
        the same hashes and steps as the source records.  No test gradients
        are involved and nothing is scored.

        Because a DVEmb score is the plain inner product ``<e, g_test>``,
        attribution then reduces to TracIn over the stored embeddings::

            (train_dir, test_dir), = attr.cache(train_ds, test_ds)   # or your own
            dvemb_dir = attr.cache_dvemb(train_dir)                   # hooked run
            scores = TracInAttributor(args).attribute_from_cache(dvemb_dir, test_dir)

        and *any* later test set (its gradients collected at the final model
        ``theta_T``) can be scored the same way without re-sweeping the trajectory.

        Args:
            train_gradients_dir: Per-step train gradient store (supplies both
                the embedded records and each step's Fisher factor).
            dvemb_dir: Where to store the embeddings; defaults to
                ``<args.output_dir>/dvemb_grads``.
            selected_training_steps: Restrict which steps' embeddings are
                *stored* (the sweep always propagates through every step).
            hessian_mode: As in :meth:`attribute_from_cache`.
            final_step: As in :meth:`attribute_from_cache`.
            loss_reduction: As in :meth:`attribute_from_cache`.
            verbose: As in :meth:`attribute_from_cache`.
            layer_name: As in :meth:`attribute_from_cache`.
            learning_rate: As in :meth:`attribute_from_cache`.

        Returns:
            ``dvemb_dir``.
        """
        if dvemb_dir is None:
            dvemb_dir = str(pathlib.Path(self.args.output_dir) / "dvemb_grads")
        self._validate(
            loss_reduction,
            "train",
            hessian_mode,
            loop_over_test=False,
            dvemb_dir=None,
        )
        train_store = GradientStorageManager(train_gradients_dir)
        prop_steps, output_steps, _final, learning_rate = self._resolve_sweep(
            train_store,
            self._read_lr_schedule(train_gradients_dir),
            selected_training_steps,
            final_step,
            learning_rate,
        )
        train_source = self.load_train_rep(
            train_store,
            steps=prop_steps,
            layer_name=layer_name,
        )
        embeddings = self.embed_trajectory(
            train_source,
            prop_steps,
            output_steps,
            learning_rate=learning_rate,
            loss_reduction=loss_reduction,
            hessian_mode=hessian_mode,
            verbose=verbose,
        )
        self.cache_representations(
            embeddings,
            GradientStorageManager(dvemb_dir),
            transform=lambda block: block,  # already the final representation
            sample_id_key=train_store.sample_id_key,
        )
        return dvemb_dir

    # ------------------------------------------------------------------ #
    # Learning-rate schedule                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_lr(learning_rate: LearningRate) -> float | dict[int, float]:
        """Validate / canonicalise a per-attribution learning-rate schedule."""
        if isinstance(learning_rate, Mapping):
            return {int(k): float(v) for k, v in learning_rate.items()}
        lr = float(learning_rate)
        if lr < 0:
            raise ValueError(f"learning_rate must be non-negative, got {lr}.")
        return lr

    @staticmethod
    def _lr(learning_rate: float | dict[int, float], step: int) -> float:
        """Learning rate ``eta`` at *step*."""
        if isinstance(learning_rate, dict):
            try:
                return learning_rate[step]
            except KeyError:
                raise ValueError(
                    f"learning_rate mapping has no entry for step {step}; it must "
                    f"cover every propagated step (step < final_step). "
                    f"Provided steps: {sorted(learning_rate)}.",
                ) from None
        return learning_rate

    @staticmethod
    def _write_lr_schedule(train_gradients_dir: str, lrs: Mapping[int, float]) -> None:
        """Persist the per-step LR actually applied during training."""
        root = pathlib.Path(train_gradients_dir)
        root.mkdir(exist_ok=True, parents=True)
        with (root / _LR_SCHEDULE_FILE).open("w", encoding="utf-8") as f:
            json.dump({str(k): float(v) for k, v in lrs.items()}, f)

    @staticmethod
    def _read_lr_schedule(train_gradients_dir: str) -> dict[int, float] | None:
        """The per-step LR recorded by :meth:`cache`, or ``None`` if absent (e.g.
        a directory produced outside the on-the-fly workflow).
        """
        path = pathlib.Path(train_gradients_dir) / _LR_SCHEDULE_FILE
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return {int(k): float(v) for k, v in json.load(f).items()}

    def _warn_on_lr_mismatch(
        self,
        learning_rate: float | dict[int, float],
        recorded_lr: Mapping[int, float],
        prop_steps: list[int],
    ) -> None:
        """Warn if the configured ``learning_rate`` disagrees with the recorded
        training schedule over any propagated step.  The configured schedule is
        still the one used for the Fisher factors ``(I - eta H)`` -- this only flags
        a likely mismatch with the trajectory that produced the gradients.
        """
        mismatched: list[tuple[int, float, float]] = []
        for s in sorted(prop_steps):
            want = recorded_lr.get(s)
            if want is None:
                continue
            try:
                got = self._lr(learning_rate, s)
            except ValueError:
                continue  # configured schedule missing this step; surfaces later
            if abs(got - want) > 1e-9 + 1e-6 * abs(want):
                mismatched.append((s, got, want))
        if mismatched:
            s0, g0, w0 = mismatched[0]
            warnings.warn(
                f"DVEmb learning_rate disagrees with the schedule recorded during "
                f"training at {len(mismatched)}/{len(prop_steps)} step(s) (e.g. "
                f"step {s0}: configured {g0:g} vs recorded {w0:g}). The configured "
                f"schedule is used, but the Fisher factors (I - eta H) will not match "
                f"the trajectory -- set learning_rate to the recorded schedule.",
                stacklevel=2,
            )

    def _resolve_sweep(
        self,
        train_store: GradientStorageManager,
        recorded_lr: dict[int, float] | None,
        selected_training_steps: Iterable[int] | None,
        final_step: int | None,
        learning_rate: LearningRate,
    ) -> tuple[list[int], set, int, float | dict[int, float]]:
        """Resolve the sweep parameters shared by scoring and embedding-only runs.

        Fixes ``final_step`` (default: one past the last recorded step in
        *train_store*), derives the propagated steps (< ``final_step``) and the
        emitted subset (``selected_training_steps`` filters rows, never the
        Fisher product), and canonicalises ``learning_rate`` -- warning when it
        disagrees with *recorded_lr*.

        Returns ``(prop_steps, output_steps, final_step, learning_rate)``.
        """
        available = train_store.available_steps()
        if final_step is None:
            final_step = (max(available) + 1) if available else 0
        prop_steps = [s for s in available if s < final_step]
        if not prop_steps:
            raise ValueError(
                f"No training step satisfies step < final_step ({final_step}); "
                f"available steps: {available}.",
            )
        learning_rate = self._normalize_lr(learning_rate)
        if recorded_lr is not None:
            self._warn_on_lr_mismatch(learning_rate, recorded_lr, prop_steps)
        if selected_training_steps is None:
            output_steps = set(prop_steps)
        else:
            output_steps = set(
                resolve_steps(train_store, selected_training_steps)
            ) & set(
                prop_steps,
            )
        return prop_steps, output_steps, final_step, learning_rate

    @staticmethod
    def _validate(
        loss_reduction: str,
        propagation: str,
        hessian_mode: str,
        loop_over_test: bool,
        dvemb_dir: str | None,
    ) -> None:
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}.",
            )
        if propagation not in ("test", "train"):
            raise ValueError(
                f"propagation must be 'test' or 'train', got {propagation!r}.",
            )
        if hessian_mode not in ("full", "diagonal"):
            raise ValueError(
                f"hessian_mode must be 'full' or 'diagonal', got {hessian_mode!r}.",
            )
        if propagation == "train" and loop_over_test:
            raise ValueError(
                "loop_over_test applies only to propagation='test': the "
                "train-side sweep is test-independent (its memory is the (d, d) "
                "operator, not the test embedding), so there is nothing to "
                "block over.",
            )
        if dvemb_dir is not None and propagation != "train":
            raise ValueError(
                "dvemb_dir requires propagation='train': data value embeddings "
                "only exist in the train-side sweep.",
            )

    # ------------------------------------------------------------------ #
    # The two sweeps                                                       #
    # ------------------------------------------------------------------ #

    def _steps_bar(self, prop_steps: list[int], desc: str, verbose: bool) -> Iterable:
        return tqdm(
            sorted(prop_steps, reverse=True),
            desc=desc,
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose or not self.args.should_log,
        )

    def _layer_slices(
        self,
        train_source: DiskGradientSource,
        step: int,
    ) -> tuple[list[str], dict[str, tuple[int, int]]]:
        """Layer order and concatenated-axis slices, peeked from one train block."""
        for _s, train_g, _hashes in train_source.for_steps([step]):
            mat = _dense_float(train_g.to(self.args.device))
            layers = list(mat.data)
            slices: dict[str, tuple[int, int]] = {}
            offset = 0
            for name in layers:
                slices[name] = (offset, offset + mat.data[name].shape[1])
                offset = slices[name][1]
            return layers, slices
        raise ValueError(f"No train gradients recorded at step {step}.")

    def embed_trajectory(
        self,
        train_source: DiskGradientSource,
        prop_steps: list[int],
        output_steps: set,
        *,
        learning_rate: float | dict[int, float],
        loss_reduction: str = "mean",
        hessian_mode: str = "full",
        verbose: bool = False,
    ) -> Iterator[StreamBlock]:
        """The train-side sweep: turn the recorded trajectory into **data value
        embeddings**, one block at a time.

        Sweeps ``prop_steps`` latest->earliest carrying the accumulated operator
        ``M`` (``I - M`` is the transposed Fisher product of all later steps)
        over the **concatenated** layer dimension ``d = sum d_layer``, and
        yields each emitted step's blocks as ``(step, Gradient, hashes)`` with
        the embeddings ``e = eta (g_hat - M g_hat)`` stored dense per layer
        (eta folded in; same hashes and step as the source records).  After a
        step's blocks are embedded, ``M`` is advanced by that step's full Fisher
        factor ``M <- M + eta * scale * sum_b (g_hat_b - M g_hat_b) g_hat_b^T``.

        The yielded blocks are an ordinary (single-shot) ``GradientSource``:
        scoring them against final-model test gradients is the inherited
        inner-product loop, and persisting them is :meth:`cache_representations`.

        Args:
            train_source: Source of the recorded train blocks (random access
                by step through :meth:`DiskGradientSource.for_steps`).
            prop_steps: Steps the sweep propagates through.
            output_steps: Steps whose embeddings are yielded.
            learning_rate: As in :meth:`attribute_from_cache`.
            loss_reduction: As in :meth:`attribute_from_cache`.
            hessian_mode: ``"full"`` (dense concatenated operator) or
                ``"diagonal"`` (one ``(d_l, d_l)`` block per layer, block-
                diagonal across layers -- the reference implementation).
            verbose: Show the per-step progress bar.

        Yields:
            ``(step, Gradient, hashes)`` embedding blocks, latest step first.
        """
        device = self.args.device
        layers, slices = self._layer_slices(train_source, max(prop_steps))
        d_total = slices[layers[-1]][1] if layers else 0
        if hessian_mode == "full":
            op_elems, op_shape = d_total * d_total, f"({d_total}, {d_total})"
        else:
            op_elems = sum((e - s) ** 2 for s, e in slices.values())
            op_shape = "block-diagonal"
        gib = op_elems * 4 / 2**30
        if gib > 2.0:
            warnings.warn(
                f"propagation='train' maintains a {op_shape} float32 "
                f"operator (~{gib:.1f} GiB) on {device}. Project the "
                "gradients at collection time and/or restrict layer_name "
                "to shrink d, or use propagation='test'.",
                stacklevel=3,
            )
        M: torch.Tensor | None = None  # lazy: no (d, d) alloc for the last step
        M_blocks: dict[str, torch.Tensor | None] = dict.fromkeys(layers)
        for ts in self._steps_bar(prop_steps, "DVEmb: embedding (train side)", verbose):
            lr = self._lr(learning_rate, ts)
            emit = ts in output_steps
            # Accumulate this step's Fisher contribution across all its blocks
            # before advancing M -- every embedding of the step must use the
            # pre-step operator, and the whole batch forms one (I - eta H_ts).
            delta: torch.Tensor | None = None
            delta_blocks: dict[str, torch.Tensor] = {}
            n_t = 0
            for _s, train_g, train_hashes in train_source.for_steps([ts]):
                mat = _dense_float(train_g.to(device))
                shared = [n for n in layers if n in mat.data]
                if not shared:
                    continue
                batch = mat.data[shared[0]].shape[0]
                n_t += batch
                g_flat = torch.zeros(batch, d_total, device=device)
                for name in shared:
                    s, e = slices[name]
                    g_flat[:, s:e] = mat.data[name]
                # e_raw[b] = (I - M) g_hat_b; the embedding is eta * e_raw.
                if hessian_mode == "full":
                    e_raw = g_flat if M is None else g_flat - g_flat @ M.T
                else:
                    e_raw = g_flat.clone()
                    for name in shared:
                        Mb = M_blocks[name]
                        if Mb is not None:
                            s, e = slices[name]
                            e_raw[:, s:e] = g_flat[:, s:e] - g_flat[:, s:e] @ Mb.T
                if emit:
                    emb = lr * e_raw
                    # ``contiguous`` detaches each slice from the flat backing
                    # storage so a stored record serialises only its own layer.
                    yield (
                        ts,
                        Gradient(
                            representation=dict.fromkeys(shared, "materialized"),
                            data={
                                n: emb[:, slices[n][0] : slices[n][1]].contiguous()
                                for n in shared
                            },
                            layer_types={n: train_g.layer_types[n] for n in shared},
                        ),
                        list(train_hashes),
                    )
                if hessian_mode == "full":
                    upd = e_raw.T @ g_flat  # sum_b (I - M) g_hat_b g_hat_b^T
                    delta = upd if delta is None else delta + upd
                else:
                    for name in shared:
                        s, e = slices[name]
                        upd = e_raw[:, s:e].T @ g_flat[:, s:e]  # (d_l, d_l)
                        delta_blocks[name] = (
                            upd
                            if name not in delta_blocks
                            else delta_blocks[name] + upd
                        )
            # H_t = (1/c) sum g_hat g_hat^T: xB_t for mean-loss-recorded grads,
            # x1 for sum.
            fisher_scale = float(n_t) if loss_reduction == "mean" else 1.0
            if delta is not None:
                scaled = (lr * fisher_scale) * delta
                M = scaled if M is None else M + scaled
            for name, upd in delta_blocks.items():
                scaled = (lr * fisher_scale) * upd
                M_blocks[name] = (
                    scaled if M_blocks[name] is None else M_blocks[name] + scaled
                )

    def _collect_test_matrix(
        self,
        test_source: DiskGradientSource,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Materialise every test gradient into one dense per-layer embedding.

        Returns ``(w, test_ids)`` where ``w`` maps each layer to its
        ``(num_test, d_layer)`` final-model test gradients, rows ordered by
        first appearance of each test hash (duplicate-hash rows collapse via
        ``index_copy_`` -- last wins).
        """
        device = self.args.device
        test_ids: list[str] = []
        test_index: dict[str, int] = {}
        pending: list[tuple[Gradient, list[int]]] = []
        for _step, test_g, test_hashes in test_source:
            mat = _dense_float(test_g.to(device))
            cols: list[int] = []
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
                cols.append(test_index[h])
            pending.append((mat, cols))
        num_test = len(test_ids)
        layers = list(pending[0][0].data) if pending else []
        w: dict[str, torch.Tensor] = {
            name: torch.zeros(
                num_test, pending[0][0].data[name].shape[1], device=device
            )
            for name in layers
        }
        for mat, cols in pending:
            idx = torch.as_tensor(cols, device=device)
            for name in layers:
                w[name].index_copy_(0, idx, mat.data[name])
        return w, test_ids

    def _propagate_test_and_score(
        self,
        w: dict[str, torch.Tensor],
        n_cols: int,
        train_source: DiskGradientSource,
        prop_steps: list[int],
        output_steps: set,
        *,
        learning_rate: float | dict[int, float],
        loss_reduction: str,
        hessian_mode: str,
        verbose: bool,
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """The test-side sweep for the ``n_cols`` columns held in ``w``.

        The step is the outer loop; each step's train blocks are pulled from
        ``train_source`` and materialised once **per call**.  ``w[name]`` (shape
        ``(n_cols, d)``) is scored against the step's train gradients and then
        advanced in place by that step's Fisher factor.  Because every test
        column propagates independently, scoring a subset of columns gives
        identical values to scoring them all -- this is what makes the
        ``loop_over_test`` column-blocking exact.

        Returns ``(scores (num_rows, n_cols), row_train_ids, row_steps)``.
        """
        device = self.args.device
        layers = list(w)
        row_chunks: list[torch.Tensor] = []
        row_train_ids: list[str] = []
        row_steps: list[int] = []
        for ts in self._steps_bar(
            prop_steps, "DVEmb: propagating (test side)", verbose
        ):
            lr = self._lr(learning_rate, ts)
            emit = ts in output_steps
            delta: dict[str, torch.Tensor] = {
                name: torch.zeros_like(w[name]) for name in layers
            }
            n_t = 0
            for _s, train_g, train_hashes in train_source.for_steps([ts]):
                mat = _dense_float(train_g.to(device)).data
                shared = [n for n in layers if n in mat]
                if not shared:
                    continue
                batch = mat[shared[0]].shape[0]
                n_t += batch
                # D[i, j] = <g(z*_i), w_j> summed over layers -> (B, n_cols).
                D_layer = {name: mat[name] @ w[name].T for name in shared}
                D = torch.zeros(batch, n_cols, device=device)
                for name in shared:
                    D += D_layer[name]
                if emit:
                    row_chunks.append((lr * D).detach().to("cpu", torch.float))
                    row_train_ids.extend(train_hashes)
                    row_steps.extend([ts] * batch)
                # Fisher update term: sum_i D[i, j] g(z*_i) -> (n_cols, d).
                # "full" drives every layer's update with the whole-model
                # alignment D (cross-layer H blocks); "diagonal" uses each
                # layer's own alignment only (block-diagonal H).
                for name in shared:
                    src = D if hessian_mode == "full" else D_layer[name]
                    delta[name] += src.T @ mat[name]
            fisher_scale = float(n_t) if loss_reduction == "mean" else 1.0
            for name in layers:
                w[name] -= lr * fisher_scale * delta[name]
        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, n_cols, dtype=torch.float)
        )
        return scores, row_train_ids, row_steps

    # ------------------------------------------------------------------ #
    # Entry points                                                         #
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
        verbose: bool = False,
        loop_over_test: bool = False,
        gradient_cache_residency: str = "disk",
        propagation: str = "train",
        dvemb_dir: str | None = None,
        hessian_mode: str = "full",
        selected_training_steps: Iterable[int] | None = None,
        loss_reduction: str = "mean",
        learning_rate: LearningRate = 1.0,
    ) -> AttributionScore:
        """Score **on the fly**: collect the trajectory, then attribute from it.

        Equivalent to :meth:`cache` (live train trajectory + final-model
        ``theta_T`` test gradients) followed by :meth:`attribute_from_cache`.
        ``final_step`` is not exposed here -- it is the number of training steps
        just run.  DVEmb always collects the trajectory once (it is a
        single-shot training pass), so ``gradient_cache_residency`` only
        relocates that store: ``"disk"`` (default) persists it under
        ``args.output_dir``; ``"memory"`` / ``"tiered"`` keep it in RAM and
        release it on return.

        The ``learning_rate`` schedule used for the Fisher product and
        ``loss_reduction`` should match the live training run configured by
        ``args`` (e.g. for a constant schedule, set
        ``learning_rate == args.learning_rate``).

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            hook_config: As in :meth:`cache`.
            verbose: As in :meth:`attribute_from_cache`.
            loop_over_test: As in :meth:`attribute_from_cache`.
            gradient_cache_residency: Where the collected trajectory lives.
            propagation: As in :meth:`attribute_from_cache`.
            dvemb_dir: As in :meth:`attribute_from_cache`.
            hessian_mode: As in :meth:`attribute_from_cache`.
            selected_training_steps: As in :meth:`attribute_from_cache`.
            loss_reduction: As in :meth:`attribute_from_cache`.
            learning_rate: As in :meth:`attribute_from_cache`.
        """
        self.require_task("attribute")
        if gradient_cache_residency not in CACHE_RESIDENCIES:
            raise ValueError(
                "gradient_cache_residency must be one of "
                f"{list(CACHE_RESIDENCIES)}, got {gradient_cache_residency!r}.",
            )
        options = {
            "propagation": propagation,
            "dvemb_dir": dvemb_dir,
            "hessian_mode": hessian_mode,
            "loop_over_test": loop_over_test,
            "selected_training_steps": selected_training_steps,
            "loss_reduction": loss_reduction,
            "verbose": verbose,
            "learning_rate": learning_rate,
        }
        if gradient_cache_residency == "disk":
            ((train_dir, test_dir),) = self.cache(
                train_dataset,
                test_dataset,
                hook_config=hook_config,
            )
            return self.attribute_from_cache(train_dir, test_dir, **options)
        # In-RAM residency: collect the trajectory + test into ephemeral stores
        # (context-managed so a tiered spill is cleaned up) and sweep them
        # directly -- the recorded LR schedule stays in memory.
        with (
            GradientStorageManager(
                tempfile.mkdtemp(prefix="dvemb_train_"),
                residency=gradient_cache_residency,
            ) as train_store,
            GradientStorageManager(
                tempfile.mkdtemp(prefix="dvemb_test_"),
                residency=gradient_cache_residency,
            ) as test_store,
        ):
            recorded_lr = self.collect_trajectory(
                train_store,
                test_store,
                train_dataset,
                test_dataset,
                hook_config=hook_config,
            )
            return self.attribute_from_cache(
                train_store,
                test_store,
                recorded_lr=recorded_lr,
                algorithm_meta={"gradient_cache_residency": gradient_cache_residency},
                **options,
            )

    def attribute_from_cache(
        self,
        train_source: str | pathlib.Path | GradientStorageManager | DiskGradientSource,
        test_source: str | pathlib.Path | GradientStorageManager | DiskGradientSource,
        *,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        loop_over_test: bool = False,
        algorithm_meta: dict | None = None,
        propagation: str = "train",
        dvemb_dir: str | None = None,
        hessian_mode: str = "full",
        final_step: int | None = None,
        loss_reduction: str = "mean",
        learning_rate: LearningRate = 1.0,
        recorded_lr: dict[int, float] | None = None,
    ) -> AttributionScore:
        """Compute the ``(num_train_rows, num_test)`` DVEmb score from collected
        gradients.

        Args:
            train_source: Per-step train gradients (directory, open store, or
                source).  Supplies both the scored train gradients and, at
                every step, the per-sample gradients forming that step's
                Fisher factor.
            test_source: Test gradients collected at the **final** model
                ``theta_T`` (the score dots against ``dl(theta_T, z_val)``).
            selected_training_steps: Restrict which training steps become
                output **rows**; ``None`` emits every step ``< final_step``.
                The propagation product always uses *every* step
                ``< final_step`` -- this only selects which rows are reported.
            layer_name: Restrict scoring (and the per-step Fisher) to this
                subset of the *stored* layers.
            verbose: Show tqdm progress bars on the logging process.
            loop_over_test: ``propagation="test"`` only.  ``False`` (default)
                holds one dense test embedding and streams the training
                gradients exactly **once**; ``True`` gives each test block its
                own sweep, re-streaming the training gradients per block, to
                bound memory to one block's embedding.  Identical scores.
            algorithm_meta: Extra entries for the score's metadata.
            propagation: Which side carries the Fisher product -- ``"train"``
                (default; the data value embeddings, scored by the inherited
                inner-product loop; its ``d^2`` operator makes it practical
                only for low-dimensional -- projected -- gradients) or
                ``"test"`` (matrix-free; memory scales with ``num_test x d``).
            dvemb_dir: Requires ``propagation="train"``.  When given, the
                embeddings computed during the sweep are also persisted there
                (see :meth:`cache_dvemb`).
            hessian_mode: ``"full"`` (default) -- the exact rank-1 sum over
                concatenated per-sample gradients, whose off-diagonal blocks
                couple layers -- or ``"diagonal"`` (block-diagonal per layer,
                the official implementation's approximation).
            final_step: Capital ``T``: the step index of the final model the
                test gradients were taken at.  ``None`` (default) uses
                ``max(available step) + 1``.
            loss_reduction: How the *training* loss was reduced over each
                minibatch -- ``"mean"`` (default) or ``"sum"``; fixes the scale
                of the empirical Fisher ``H_t`` (see the module docstring).
            learning_rate: The SGD learning-rate schedule ``eta`` -- a float or
                ``{step: eta_step}`` covering every propagated step.  A
                mismatch with the recorded schedule warns.
            recorded_lr: The schedule recorded at collection; ``None`` reads
                ``lr_schedule.json`` beside the train store if present.

        Returns:
            An :class:`AttributionScore`; also persisted to ``args.output_dir``.
        """
        self._validate(
            loss_reduction,
            propagation,
            hessian_mode,
            loop_over_test=loop_over_test,
            dvemb_dir=dvemb_dir,
        )
        train_store = self.resolve_store(train_source)
        test_store = self.resolve_store(test_source)
        if recorded_lr is None:
            recorded_lr = self._read_lr_schedule(str(train_store.save_dir))
        prop_steps, output_steps, final_step, learning_rate = self._resolve_sweep(
            train_store,
            recorded_lr,
            selected_training_steps,
            final_step,
            learning_rate,
        )
        # The train source is restricted to the propagated steps; the sweep
        # pulls one step at a time from it.  The test source supplies every
        # column (the final-model gradients).
        train = self.load_train_rep(
            train_store, steps=prop_steps, layer_name=layer_name
        )
        test = self.load_test_rep(test_store, layer_name=layer_name, verbose=verbose)
        sweep = {
            "learning_rate": learning_rate,
            "loss_reduction": loss_reduction,
            "hessian_mode": hessian_mode,
            "verbose": verbose,
        }
        if propagation == "train":
            embeddings = self.embed_trajectory(
                train,
                prop_steps,
                output_steps,
                **sweep,
            )
            if dvemb_dir is not None:
                embeddings = self._tee_to_store(
                    embeddings,
                    GradientStorageManager(dvemb_dir),
                    train_store.sample_id_key,
                )
            # The embeddings are an ordinary single-shot source; the score is
            # the inherited inner product against the dense test gradients.
            scores, row_train_ids, row_steps, test_ids = self.score_sources(
                embeddings,
                test,
                transform_test=_dense_float,
            )
        elif not loop_over_test:
            # Step outer / test inner: one dense embedding, train read once.
            w, test_ids = self._collect_test_matrix(test)
            scores, row_train_ids, row_steps = self._propagate_test_and_score(
                w,
                len(test_ids),
                train,
                prop_steps,
                output_steps,
                **sweep,
            )
        else:
            # Test outer: one block's embedding resident, train re-streamed per
            # block.  Pass 1 fixes the column order from hashes alone.
            test_ids, test_index = [], {}
            for _step, _tg, test_hashes in test:
                for h in test_hashes:
                    if h not in test_index:
                        test_index[h] = len(test_ids)
                        test_ids.append(h)
            scores = None
            row_train_ids, row_steps = [], []
            for _step, test_g, test_hashes in test:
                w_block = _dense_float(test_g.to(self.args.device)).data
                block_cols = [test_index[h] for h in test_hashes]
                block_scores, rtids, rsteps = self._propagate_test_and_score(
                    w_block,
                    len(block_cols),
                    train,
                    prop_steps,
                    output_steps,
                    **sweep,
                )
                if scores is None:
                    scores = torch.zeros(block_scores.shape[0], len(test_ids))
                    row_train_ids, row_steps = rtids, rsteps
                scores[:, block_cols] = block_scores
            if scores is None:
                scores = torch.zeros(0, len(test_ids), dtype=torch.float)

        return self.build_score(
            scores,
            row_train_ids,
            row_steps,
            test_ids,
            algorithm_meta={
                "final_step": final_step,
                "selected_training_steps": sorted(output_steps),
                "propagated_steps": sorted(prop_steps),
                "learning_rate": learning_rate,
                "loss_reduction": loss_reduction,
                "propagation": propagation,
                "dvemb_dir": dvemb_dir,
                "hessian_mode": hessian_mode,
                **self.stores_meta(train_store, test_store),
                **(algorithm_meta or {}),
            },
            layer_name=train.layer_name,
        )

    @staticmethod
    def _tee_to_store(
        blocks: Iterable[StreamBlock],
        store: GradientStorageManager,
        sample_id_key: str | int | None,
    ) -> Iterator[StreamBlock]:
        """Persist each embedding block as it streams by, then pass it on.

        Yields:
            The blocks of *blocks*, unchanged.
        """
        from dattri_llm.gradient.gradient import GradientRecord

        for step, block, hashes in blocks:
            store.save_bulk(
                [
                    GradientRecord(
                        step=step,
                        input_hash=list(hashes),
                        gradient=block.to("cpu"),
                        sample_id_key=sample_id_key,
                    ),
                ],
            )
            yield step, block, hashes
