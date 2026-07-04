"""DVEmb (Data Value Embedding) trajectory-aware attribution (workflow 2).

Like :class:`~dattri_llm.attribution.algorithm.tracin.TracInAttributor` and the K-FAC
family, this attributor consumes :class:`~dattri_llm.gradient.gradient.Gradient`
records previously persisted by
:class:`~dattri_llm.gradient.file_manager.GradientFileManager`; no
forward/backward pass is run at attribution time.

Unlike TracIn — which simply dots the train gradient at the step a sample was
used against the test gradient at that *same* checkpoint — DVEmb accounts for
how a training update at step ``t_s`` keeps propagating through every
*subsequent* training step before reaching the final model ``θ_T``.  Following
"Capturing the Temporal Dependence of Training Data Influence"
(https://arxiv.org/abs/2412.09538), the influence of a training sample ``z*``
used at step ``t_s`` on a test point ``z_val`` is

    I(z*, t_s) = η_{t_s} · ∇ℓ(θ_T, z_val)ᵀ
                 [ ∏_{k=t_s+1}^{T-1} (I − η_k H_k) ] ∇ℓ(θ_{t_s}, z*)        (1)

i.e. the train gradient at ``t_s`` is pushed forward through the product of the
SGD Jacobians ``(I − η_k H_k)`` of every later step ``k`` and then dotted with
the test gradient taken at the **final** model ``θ_T`` (capital ``T`` =
``final_step``).  The per-step Hessian is the Gauss-Newton / empirical-Fisher
approximation built from that step's recorded per-sample gradients,

    H_k ≈ (1/c) Σ_{z ∈ B_k} ĝ(θ_k, z) ĝ(θ_k, z)ᵀ                            (2)

where ``ĝ`` is the *recorded* per-sample gradient and ``c`` its per-sample loss
weight: the empirical Fisher must use the **true** per-sample gradients, so when
the gradients were recorded under a **mean** loss (``ĝ = ∇ℓ / B``, ``c = 1/B``)
the sum of recorded outer products is rescaled by the step's batch size ``B``,
while under a **sum** loss (``ĝ = ∇ℓ``, ``c = 1``) it is used as-is — see the
``loss_reduction`` argument of :meth:`attribute`.

Setting every ``H_k = 0`` recovers TracIn (η · ⟨g_test, g_train⟩); the Fisher
factors are exactly the "training dynamics" correction DVEmb adds.

**Computation.** The bilinear form in (1) can be evaluated by carrying the
Fisher product on either side; the ``propagation`` argument of
:meth:`attribute` / :meth:`attribute_from_cache` selects which.  Both sweep
the recorded training steps from latest to earliest and produce **identical**
scores.

With ``propagation="test"`` (the default) the product is applied to the *test*
side: for every test column a running parameter-space vector

    w_{t_s} = [ ∏_{k=t_s+1}^{T-1}(I − η_k H_k) ]ᵀ ∇ℓ(θ_T, z_val)

is initialised at the final-model test gradient.  At each step ``t_s``
(descending) the rows for the train samples recorded there are
``η_{t_s} · ⟨g(z*), w_{t_s}⟩``, after which ``w`` is advanced by that step's
full Fisher factor ``w ← w − η_{t_s} Σ_{z∈B_{t_s}} g(z) ⟨g(z), w⟩``.  The
recorded per-sample gradients of a step thus serve twice: as the vectors
scored, and as the rank-1 components of ``H_{t_s}``.  This is matrix-free —
the product ``∏(I − η H)`` is never materialised — but ties the sweep to the
given test set.

With ``propagation="train"`` the product is applied to the *train* side,
yielding the paper's **data value embedding** per training sample,

    e_{t_s}(z*) = η_{t_s} · [ ∏_{k=t_s+1}^{T-1}(I − η_k H_k) ]ᵀ ĝ(θ_{t_s}, z*),

so a row's score against any test column is simply ``⟨e, ∇ℓ(θ_T, z_val)⟩``.
Because the propagated object now lives on the train side, the sweep carries
the explicit accumulated operator ``M_{t_s}`` with
``I − M_{t_s} = [ ∏_{k>t_s}(I − η_k H_k) ]ᵀ`` — a dense ``(d, d)`` matrix over
the concatenated layer dimension — updated as
``M ← M + η_{t_s} Σ_{z∈B_{t_s}} (ĝ(z) − M ĝ(z)) ĝ(z)ᵀ``.  This costs ``d²``
memory (use projected gradients or ``layer_name`` to keep ``d`` small) but
makes the embeddings test-independent.

This is the **basic** DVEmb estimator — it materialises the per-layer gradients
and propagates the exact (Fisher-approximated) product.  Influence-checkpointing
and the low-rank embedding compression of the paper are deliberately omitted.

The result is an :class:`~dattri_llm.attribution.score.AttributionScore` whose
rows are ``(train_hash, step)`` pairs (one row per recorded checkpoint of a
sample, stamped with its step) and whose columns are the test-sample hashes in
on-disk order — identical bookkeeping to TracIn and the K-FAC family.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dattri_llm.attribution.arguments import AttributionArguments
from dattri_llm.attribution.base import BaseAttributor
from dattri_llm.attribution.utils import normalize_layer_names, task_loss_fn
from dattri_llm.attribution.score import AttributionScore
from dattri_llm.gradient.datasets import resolve_steps
from dattri_llm.gradient.streaming import DiskGradientSource, GradientStreamer
from dattri.task import AttributionTask
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.hooks import HookManagerConfig
from dattri_llm.gradient.gradient import Gradient, GradientRecord


class DVEmbAttributor(BaseAttributor):
    """DVEmb (Data Value Embedding) attributor over pre-collected on-disk gradients.

    Scores every training record (at the step it was recorded) against every
    test record, correcting the raw TracIn inner product for how the update
    propagates through all *later* training steps up to the final model
    ``θ_T`` — see the module docstring for the score definition.

    Args:
        args: :class:`AttributionArguments` controlling DataLoader behaviour,
            device placement, and the output directory.  Only the
            ``dataloader_*`` fields, ``device``, and ``output_dir`` are
            consulted in this workflow.
        task: Required by the on-the-fly :meth:`cache` / :meth:`attribute`
            (supplies the model, the loss, and the optional ``target_func`` for
            the test side); unused by :meth:`attribute_from_cache`.

    The learning-rate schedule ``η`` is a per-attribution argument of
    :meth:`attribute` / :meth:`attribute_from_cache` (a float for a constant
    schedule or a ``{step: η_step}`` mapping); it enters both the per-step score
    scale ``η_{t_s}`` and the Fisher factors ``(I − η_k H_k)``, so it must match
    the schedule the gradients were collected under.

    Layer selection happens at **capture** (the ``hook_config`` of the live
    methods, or whatever was hooked when the cache was collected).  By default
    every stored layer enters the Fisher and the score;
    :meth:`attribute_from_cache` additionally takes a ``layer_name`` read-time
    filter to score a subset of the stored layers.
    """

    algorithm = "DVEmb"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: Optional[AttributionTask] = None,
    ) -> None:
        self.args = args
        self.task = task

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: Optional[str] = None,
        offload_interval: int = 1,
        hook_config: Optional[HookManagerConfig] = None,
    ) -> Tuple[str, str]:
        """Run the training trajectory live and cache the gradients DVEmb needs.

        DVEmb is trajectory-aware, so — unlike the trajectory-agnostic attributors
        — it cannot collect both sides at one checkpoint.  It needs (a) the
        per-step **train** gradients along the trajectory and (b) the **test**
        gradients at the *final* model ``θ_T``, which only exists once training
        finishes.  This method therefore:

        1. drives one live training pass from the task's first checkpoint
           (``GradientStreamer`` with ``enable_update=True``), offloading each
           step's per-batch train gradient to disk;
        2. then — with the model now advanced to ``θ_T`` — runs a frozen pass
           over the test set, offloading those gradients.

        The two directories feed straight into :meth:`attribute_from_cache`, so
        on-the-fly :meth:`attribute` is exactly *cache + attribute_from_cache*.

        Args:
            train_dataset, test_dataset: Datasets to stream (batches go straight
                to the task's loss / ``target_func``).
            cache_dir: Parent directory for the two gradient stores
                (``<cache_dir>/train_grads`` and ``<cache_dir>/test_grads``);
                defaults to ``args.output_dir``.
            offload_interval: Steps accumulated per gradient file.  ``1``
                (default) writes one file per step — best for DVEmb's per-step
                sweep (no redundant multi-step file reloads).
            hook_config: :class:`HookManagerConfig` for the internal streamers
                (which layers to hook, per-layer projection, ...).  ``None`` uses
                the streamer default (factorized hooks on every linear-family layer).

        Returns:
            ``(train_gradients_dir, test_gradients_dir)``.
        """
        if self.task is None:
            raise ValueError(
                "cache() (live collection) requires a `task` with a model; pass "
                "pre-collected gradients to attribute_from_cache() instead."
            )
        n_ckpt = len(self.task.get_checkpoints())
        if n_ckpt > 1:
            warnings.warn(
                f"DVEmb regenerates the trajectory live from checkpoint 0; the "
                f"other {n_ckpt - 1} provided checkpoint(s) are ignored.",
                stacklevel=2,
            )
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        train_dir = os.path.join(cache_dir, "train_grads")
        test_dir = os.path.join(cache_dir, "test_grads")
        train_fm = GradientFileManager(train_dir)
        test_fm = GradientFileManager(test_dir)

        self.task._load_checkpoints(0)
        model = self.task.get_model()
        train_loss = task_loss_fn(self.task.original_loss_func)
        # The test side is collected against the task's target_func (which dattri
        # defaults to the loss when none is given), at the final model θ_T.
        test_loss = task_loss_fn(self.task.original_target_func)

        # 1) Train trajectory θ_0 → θ_T: offload each step's per-batch gradient.
        train_streamer = GradientStreamer(
            model, train_dataset, self.args,
            batch_size=self.args.per_device_train_batch_size,
            enable_update=True, loss_fn=train_loss, config=hook_config,
        )
        train_streamer.hook_manager.add_callback(
            OffloadCallback(offload_interval, train_fm, recording_type="per_batch")
        )
        with train_streamer:
            for _ in train_streamer:
                pass
        # Record the LR actually applied per step, so attribute_from_cache can
        # verify the configured ``learning_rate`` matches the real trajectory.
        self._write_lr_schedule(train_dir, train_streamer.learning_rates)

        # 2) Test gradients at the now-final model θ_T (frozen probe).
        test_streamer = GradientStreamer(
            model, test_dataset, self.args,
            batch_size=self.args.per_device_eval_batch_size,
            enable_update=False, loss_fn=test_loss, config=hook_config,
        )
        test_streamer.hook_manager.add_callback(
            OffloadCallback(offload_interval, test_fm, recording_type="per_batch")
        )
        with test_streamer:
            for _ in test_streamer:
                pass

        return train_dir, test_dir

    def cache_dvemb(
        self,
        train_gradients_dir: str,
        dvemb_dir: Optional[str] = None,
        *,
        selected_training_steps: Optional[Iterable[int]] = None,
        final_step: Optional[int] = None,
        loss_reduction: str = "mean",
        verbose: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        learning_rate: Union[float, Mapping[int, float]] = 1.0,
    ) -> str:
        """Turn a stored gradient trajectory into persisted **data value embeddings**.

        One train-side sweep (``propagation="train"``) over an existing
        per-step gradient store — written by :meth:`cache` or by any training
        run wrapped with hooks — turning every train record into its embedding
        ``e = η · ∏_{k>t_s}(I − η_k H_k)ᵀ ĝ`` and storing it in ``dvemb_dir``
        as materialized per-layer :class:`Gradient` records via
        :class:`GradientFileManager` — same hashes and steps as the source
        records.  No test gradients are involved and nothing is scored.

        Because a DVEmb score is the plain inner product ``⟨e, g_test⟩``,
        attribution then reduces to TracIn over the stored embeddings::

            train_dir, test_dir = attr.cache(train_ds, test_ds)   # or your own
            dvemb_dir = attr.cache_dvemb(train_dir)               # hooked run
            scores = TracInAttributor(args).attribute_from_cache(
                train_gradients_dir=dvemb_dir, test_gradients_dir=test_dir)

        and *any* later test set (its gradients collected at the final model
        ``θ_T``) can be scored the same way without re-sweeping the trajectory.

        Args:
            train_gradients_dir: Per-step train gradient store (supplies both
                the embedded records and each step's Fisher factor).
            dvemb_dir: Where to store the embeddings; defaults to
                ``<args.output_dir>/dvemb_grads``.
            selected_training_steps: Restrict which steps' embeddings are
                *stored* (the sweep always propagates through every step);
                ``None`` stores all of them.
            final_step, loss_reduction, verbose, layer_name, learning_rate: As
                in :meth:`attribute_from_cache`.

        Returns:
            ``dvemb_dir``.
        """
        if dvemb_dir is None:
            dvemb_dir = os.path.join(self.args.output_dir, "dvemb_grads")
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}."
            )
        (
            _train_fm, prop_steps, output_steps, _final_step, learning_rate,
        ) = self._resolve_sweep(
            train_gradients_dir, selected_training_steps, final_step,
            learning_rate,
        )
        device = self.args.device
        train_source = DiskGradientSource(
            GradientFileManager(train_gradients_dir), self.args,
            steps=prop_steps, layer_name=normalize_layer_names(layer_name),
        )
        # No test matrix to derive the concatenated layout from — peek it off
        # the latest propagated step's first train block instead.
        layers, slices = self._train_layer_slices(
            train_source, max(prop_steps), device,
        )
        self._propagate_train_and_score(
            None, layers, slices, train_source, prop_steps, output_steps,
            device, loss_reduction, verbose, learning_rate,
            dvemb_fm=GradientFileManager(dvemb_dir),
        )
        return dvemb_dir

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_lr(
        learning_rate: Union[float, Mapping[int, float]],
    ) -> Union[float, Dict[int, float]]:
        """Validate / canonicalise a per-attribution learning-rate schedule."""
        if isinstance(learning_rate, Mapping):
            return {int(k): float(v) for k, v in learning_rate.items()}
        lr = float(learning_rate)
        if lr < 0:
            raise ValueError(f"learning_rate must be non-negative, got {lr}.")
        return lr

    @staticmethod
    def _lr(learning_rate: Union[float, Dict[int, float]], step: int) -> float:
        """Learning rate ``η`` at *step*."""
        if isinstance(learning_rate, dict):
            try:
                return learning_rate[step]
            except KeyError:
                raise ValueError(
                    f"learning_rate mapping has no entry for step {step}; it must "
                    f"cover every propagated step (step < final_step). "
                    f"Provided steps: {sorted(learning_rate)}."
                ) from None
        return learning_rate

    @staticmethod
    def _materialize(block: Gradient, device: torch.device) -> Dict[str, torch.Tensor]:
        """Materialise a gradient block into ``{layer: (B, d) float tensor}``."""
        mat = block.to(device).materialize()
        return {name: value.float() for name, value in mat.data.items()}

    @staticmethod
    def _lr_schedule_path(train_gradients_dir: str) -> str:
        return os.path.join(train_gradients_dir, "lr_schedule.json")

    def _write_lr_schedule(
        self, train_gradients_dir: str, lrs: Mapping[int, float]
    ) -> None:
        """Persist the per-step LR actually applied during training."""
        os.makedirs(train_gradients_dir, exist_ok=True)
        with open(self._lr_schedule_path(train_gradients_dir), "w") as f:
            json.dump({str(k): float(v) for k, v in lrs.items()}, f)

    def _read_lr_schedule(
        self, train_gradients_dir: str
    ) -> Optional[Dict[int, float]]:
        """The per-step LR recorded by :meth:`cache`, or ``None`` if absent (e.g.
        a directory produced outside the on-the-fly workflow)."""
        path = self._lr_schedule_path(train_gradients_dir)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return {int(k): float(v) for k, v in json.load(f).items()}

    def _warn_on_lr_mismatch(
        self,
        learning_rate: Union[float, Dict[int, float]],
        recorded_lr: Mapping[int, float],
        prop_steps: List[int],
    ) -> None:
        """Warn if the configured ``learning_rate`` disagrees with the recorded
        training schedule over any propagated step.  The configured schedule is
        still the one used for the Fisher factors ``(I − η H)`` — this only flags
        a likely mismatch with the trajectory that produced the gradients."""
        mismatched: List[Tuple[int, float, float]] = []
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
                f"schedule is used, but the Fisher factors (I − η H) will not match "
                f"the trajectory — set learning_rate to the recorded schedule.",
                stacklevel=2,
            )

    def _propagate_test_and_score(
        self,
        w: Dict[str, torch.Tensor],
        n_cols: int,
        layers: List[str],
        train_source: DiskGradientSource,
        prop_steps: List[int],
        output_steps: set,
        device: torch.device,
        loss_reduction: str,
        verbose: bool,
        learning_rate: Union[float, Dict[int, float]],
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """One latest→earliest sweep for the ``n_cols`` columns held in ``w``.

        The step is the outer loop; each step's train blocks are pulled from
        ``train_source`` via :meth:`DiskGradientSource.for_steps` (random-access
        by step) and materialised once **per call**.  ``w[name]`` (shape
        ``(n_cols, d)``) is scored against the step's train gradients and then
        advanced in place by that step's Fisher factor.  Because every test column
        propagates independently, scoring a subset of columns gives identical
        values to scoring them all — this is what makes the ``loop_over_test``
        column-blocking exact.

        Returns ``(scores (num_rows, n_cols), row_train_ids, row_steps)``.
        """
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        steps = tqdm(
            sorted(prop_steps, reverse=True),
            desc="DVEmb: propagating",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose or not self.args.should_log,
        )
        for ts in steps:
            lr = self._lr(learning_rate, ts)
            emit = ts in output_steps
            # Accumulate this step's Fisher contribution across all its blocks
            # before advancing w, so the whole batch B_ts forms one (I − η H_ts)
            # factor.  ``n_t`` counts the step's recorded samples for B_ts.
            delta: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(w[name]) for name in layers
            }
            n_t = 0
            for _s, train_g, train_hashes in train_source.for_steps([ts]):
                mat = self._materialize(train_g, device)
                shared = [n for n in layers if n in mat]
                if not shared:
                    continue
                batch = mat[shared[0]].shape[0]
                n_t += batch
                # D[i, j] = ⟨g(z*_i), w_j⟩ summed over layers → (B, n_cols).
                D = torch.zeros(batch, n_cols, device=device)
                for name in shared:
                    D += mat[name] @ w[name].T
                if emit:
                    row_chunks.append((lr * D).detach().to("cpu", torch.float))
                    row_train_ids.extend(train_hashes)
                    row_steps.extend([ts] * batch)
                # Fisher update term: Σ_i D[i, j] g(z*_i) → (n_cols, d).
                for name in shared:
                    delta[name] += D.T @ mat[name]
            # H_t = (1/c) Σ ĝ ĝᵀ: ×B_t for mean-loss-recorded grads, ×1 for sum.
            fisher_scale = float(n_t) if loss_reduction == "mean" else 1.0
            for name in layers:
                w[name] -= lr * fisher_scale * delta[name]

        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, n_cols, dtype=torch.float)
        )
        return scores, row_train_ids, row_steps

    def _propagate_train_and_score(
        self,
        test_flat: Optional[torch.Tensor],
        layers: List[str],
        slices: Dict[str, Tuple[int, int]],
        train_source: DiskGradientSource,
        prop_steps: List[int],
        output_steps: set,
        device: torch.device,
        loss_reduction: str,
        verbose: bool,
        learning_rate: Union[float, Dict[int, float]],
        dvemb_fm: Optional[GradientFileManager] = None,
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """One latest→earliest sweep carrying the operator on the *train* side.

        The counterpart of :meth:`_propagate_and_score` for
        ``propagation="train"``: instead of advancing per-test-column vectors,
        it maintains the accumulated operator ``M`` (``I − M`` is the transposed
        Fisher product of all later steps) over the **concatenated** layer
        dimension ``d = Σ d_layer`` and turns each step's recorded gradients
        into data value embeddings ``e = η (ĝ − M ĝ)``.  Emitted rows are
        ``e @ test_flatᵀ``; after a step's blocks are embedded, ``M`` is
        advanced by that step's full Fisher factor
        ``M ← M + η · scale · Σ_b (ĝ_b − M ĝ_b) ĝ_bᵀ``.

        Layers are concatenated (not block-diagonal): the Fisher rank-1 terms
        ``ĝ ĝᵀ`` couple layers exactly as in the test-side sweep, so the two
        propagation modes produce identical scores.

        Args:
            test_flat: ``(num_test, d)`` final-model test gradients, layer
                blocks concatenated in ``layers`` order.  ``None`` skips scoring
                (embedding-only sweep — requires ``dvemb_fm``).
            layers, slices: Layer order and the ``{name: (start, end)}``
                column ranges of each layer inside the concatenated axis.
            dvemb_fm: When given, each emitted step's embeddings ``e`` are also
                persisted through this :class:`GradientFileManager` as
                **materialized** per-layer :class:`Gradient` records (η folded
                in; same hashes and step as the source records).  A row's score
                is then a plain inner product ``⟨e, g_test⟩``, so the stored
                directory can be consumed directly by
                ``TracInAttributor.attribute_from_cache`` against the test
                gradients, reproducing this attributor's scores.

        Returns ``(scores (num_rows, num_test), row_train_ids, row_steps)``;
        ``scores`` is empty for an embedding-only sweep.
        """
        if test_flat is None and dvemb_fm is None:
            raise ValueError(
                "embedding-only sweep (test_flat=None) requires dvemb_fm."
            )
        d_total = slices[layers[-1]][1] if layers else 0
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        M: Optional[torch.Tensor] = None  # lazy: no (d, d) alloc for the last step
        steps = tqdm(
            sorted(prop_steps, reverse=True),
            desc="DVEmb: propagating (train side)",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose or not self.args.should_log,
        )
        for ts in steps:
            lr = self._lr(learning_rate, ts)
            emit = ts in output_steps
            # Accumulate this step's Fisher contribution across all its blocks
            # before advancing M — every embedding of the step must use the
            # pre-step operator, and the whole batch forms one (I − η H_ts).
            delta: Optional[torch.Tensor] = None
            n_t = 0
            for _s, train_g, train_hashes in train_source.for_steps([ts]):
                mat = self._materialize(train_g, device)
                shared = [n for n in layers if n in mat]
                if not shared:
                    continue
                batch = mat[shared[0]].shape[0]
                n_t += batch
                g_flat = torch.zeros(batch, d_total, device=device)
                for name in shared:
                    s, e = slices[name]
                    g_flat[:, s:e] = mat[name]
                # e_raw[b] = (I − M) ĝ_b; the embedding is η · e_raw.
                e_raw = g_flat if M is None else g_flat - g_flat @ M.T
                if emit:
                    emb = lr * e_raw
                    if test_flat is not None:
                        row_chunks.append(
                            (emb @ test_flat.T).detach().to("cpu", torch.float)
                        )
                        row_train_ids.extend(train_hashes)
                        row_steps.extend([ts] * batch)
                    if dvemb_fm is not None:
                        # Persist as an ordinary materialized per-layer Gradient
                        # record so downstream dot-product attributors (TracIn)
                        # can read it like any stored gradient.  ``contiguous``
                        # detaches the slice from the flat backing storage so
                        # torch.save doesn't serialise the whole (B, d) tensor
                        # per layer.
                        grad = Gradient(
                            representation={n: "materialized" for n in shared},
                            data={
                                n: emb[:, slices[n][0]:slices[n][1]]
                                .contiguous().cpu()
                                for n in shared
                            },
                            layer_types={
                                n: train_g.layer_types[n] for n in shared
                            },
                        )
                        dvemb_fm.save_bulk(
                            [GradientRecord(
                                step=ts, input_hash=list(train_hashes),
                                gradient=grad,
                            )]
                        )
                upd = e_raw.T @ g_flat  # Σ_b (I − M) ĝ_b ĝ_bᵀ → (d, d)
                delta = upd if delta is None else delta + upd
            if delta is not None:
                # H_t = (1/c) Σ ĝ ĝᵀ: ×B_t for mean-loss-recorded grads, ×1 for sum.
                fisher_scale = float(n_t) if loss_reduction == "mean" else 1.0
                scaled = (lr * fisher_scale) * delta
                M = scaled if M is None else M + scaled

        n_cols = test_flat.shape[0] if test_flat is not None else 0
        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, n_cols, dtype=torch.float)
        )
        return scores, row_train_ids, row_steps

    def _resolve_sweep(
        self,
        train_gradients_dir: str,
        selected_training_steps: Optional[Iterable[int]],
        final_step: Optional[int],
        learning_rate: Union[float, Mapping[int, float]],
    ) -> Tuple[GradientFileManager, List[int], set, int, Union[float, Dict[int, float]]]:
        """Resolve the sweep parameters shared by scoring and embedding-only runs.

        Opens the train store, fixes ``final_step`` (default: one past the last
        recorded step), derives the propagated steps (< ``final_step``) and the
        emitted subset (``selected_training_steps`` filters rows, never the
        Fisher product), and canonicalises ``learning_rate`` — warning when it
        disagrees with the schedule :meth:`cache` recorded.

        Returns ``(train_fm, prop_steps, output_steps, final_step, learning_rate)``.
        """
        train_fm = GradientFileManager(train_gradients_dir)
        available = train_fm.available_steps()
        if final_step is None:
            final_step = (max(available) + 1) if available else 0
        # Steps that participate in the trajectory (rows *and* Fisher product).
        prop_steps = [s for s in available if s < final_step]
        if not prop_steps:
            raise ValueError(
                f"No training step satisfies step < final_step ({final_step}); "
                f"available steps: {available}."
            )
        # If the train dir carries a schedule recorded by cache(), check the
        # configured learning_rate against it (warn-only; configured one is used).
        learning_rate = self._normalize_lr(learning_rate)
        recorded_lr = self._read_lr_schedule(train_gradients_dir)
        if recorded_lr is not None:
            self._warn_on_lr_mismatch(learning_rate, recorded_lr, prop_steps)
        # ``selected_training_steps`` filters only which steps are emitted as
        # rows; the propagation always sweeps every step in ``prop_steps``.
        if selected_training_steps is None:
            output_steps = set(prop_steps)
        else:
            output_steps = set(resolve_steps(train_fm, selected_training_steps)) & set(
                prop_steps
            )
        return train_fm, prop_steps, output_steps, final_step, learning_rate

    def _train_layer_slices(
        self, train_source: DiskGradientSource, step: int, device: torch.device
    ) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
        """Layer order and concatenated-axis slices, peeked from one train block.

        Materialises the first block of *step* to learn the stored layers and
        their materialized widths — used by the embedding-only sweep, which has
        no test matrix to derive the layout from.
        """
        for _s, train_g, _hashes in train_source.for_steps([step]):
            mat = self._materialize(train_g, device)
            layers = list(mat.keys())
            slices: Dict[str, Tuple[int, int]] = {}
            offset = 0
            for name in layers:
                slices[name] = (offset, offset + mat[name].shape[1])
                offset = slices[name][1]
            return layers, slices
        raise ValueError(f"No train gradients recorded at step {step}.")

    # ------------------------------------------------------------------ #
    # Main entry points                                                  #
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        propagation: str = "train",
        dvemb_dir: Optional[str] = None,
        loop_over_test: bool = False,
        selected_training_steps: Optional[Iterable[int]] = None,
        loss_reduction: str = "mean",
        verbose: bool = False,
        hook_config: Optional[HookManagerConfig] = None,
        learning_rate: Union[float, Mapping[int, float]] = 1.0,
    ) -> AttributionScore:
        """Score **on the fly**: cache the trajectory, then attribute from cache.

        Exactly :meth:`cache` (live train trajectory + final-model ``θ_T`` test
        gradients) followed by :meth:`attribute_from_cache`.  ``final_step`` is
        not exposed here — it is the number of training steps just run.

        The ``learning_rate`` schedule used for the Fisher product and
        ``loss_reduction`` should match the live training run configured by
        ``args`` (e.g. for a constant schedule, set
        ``learning_rate == args.learning_rate``); otherwise the propagation
        factors ``(I − η H_k)`` will not match the trajectory.  :meth:`cache`
        records the *actual* per-step LR and :meth:`attribute_from_cache` warns
        if the configured schedule disagrees with it.

        Args:
            train_dataset, test_dataset: Datasets to stream.
            propagation, dvemb_dir, loop_over_test, selected_training_steps,
                loss_reduction, verbose, learning_rate: As in
                :meth:`attribute_from_cache`.
            hook_config: As in :meth:`cache`.
        """
        train_dir, test_dir = self.cache(train_dataset, test_dataset,
                                         hook_config=hook_config)
        return self.attribute_from_cache(
            train_dir,
            test_dir,
            propagation=propagation,
            dvemb_dir=dvemb_dir,
            loop_over_test=loop_over_test,
            selected_training_steps=selected_training_steps,
            loss_reduction=loss_reduction,
            verbose=verbose,
            learning_rate=learning_rate,
        )

    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        propagation: str = "train",
        dvemb_dir: Optional[str] = None,
        loop_over_test: bool = False,
        selected_training_steps: Optional[Iterable[int]] = None,
        final_step: Optional[int] = None,
        loss_reduction: str = "mean",
        verbose: bool = False,
        layer_name: Optional[Union[str, List[str]]] = None,
        learning_rate: Union[float, Mapping[int, float]] = 1.0,
    ) -> AttributionScore:
        """Compute the ``(num_train_rows, num_test)`` DVEmb attribution score.

        Args:
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during training.  Supplies both the
                scored train gradients and, at every step, the per-sample
                gradients forming that step's Fisher factor.
            test_gradients_dir: Directory written during the test pass.  These
                gradients must have been collected at the **final** model
                ``θ_T`` (capital ``T`` = ``final_step``), since the score dots
                against ``∇ℓ(θ_T, z_val)``.
            propagation: Which side of the bilinear form carries the Fisher
                product — ``"test"`` (default) or ``"train"``.  Both produce
                **identical** scores; they differ in cost shape and reusability
                (see the module docstring for the two recursions).  ``"test"``
                back-propagates one vector per test column (matrix-free; memory
                scales with ``num_test × d``) — fastest when the test set is
                the only one that will ever be scored.  ``"train"`` is the
                paper's *data value embedding* formulation: it maintains the
                explicit ``(d, d)`` propagation operator over the concatenated
                layer dimension and turns every training record into a
                test-independent embedding, scored against the test gradients
                by a plain inner product.  Its ``d²`` operator makes it
                practical only for low-dimensional gradients — project at
                collection time (``hook_config``) and/or restrict
                ``layer_name`` to keep ``d`` small.
            dvemb_dir: Requires ``propagation="train"``.  When given, the data
                value embeddings computed during the sweep are also persisted
                to this directory (managed by :class:`GradientFileManager`) as
                materialized per-layer :class:`Gradient` records — η folded in,
                same hashes/steps as the train records, one record per emitted
                step block.  Because a row's score is then the plain inner
                product ``⟨e, g_test⟩``, the stored directory can later be
                scored against *any* test gradient directory with
                ``TracInAttributor.attribute_from_cache(dvemb_dir, test_dir)``
                — no re-sweep of the trajectory.
            loop_over_test: Memory/disk trade-off for the per-test-column
                embedding ``w`` (shape ``(num_test, d)``), which is
                back-propagated through the steps (``propagation="test"``
                only; incompatible with ``"train"``, whose sweep is
                test-independent and holds the test matrix just for the final
                inner products).  Because every test column propagates
                independently, ``False`` (default) uses the test columns as
                the **inner** loop and the steps as the outer loop: one dense
                embedding is held and the training gradients are streamed
                exactly **once** (peak memory: full ``w`` + one train block;
                fastest).  ``True`` uses the test blocks as the **outer** loop
                to save space: each test block gets its own sweep, holding
                only that block's embedding while **re-streaming the training
                gradients once per block** (peak memory: one test block's ``w`` +
                one train block).  Both produce identical scores; use ``True``
                when the full test embedding does not fit in memory.
            selected_training_steps: Restrict which training checkpoints become
                output **rows** to these steps; ``None`` (default) emits a row
                for every step ``< final_step``.  The propagation product always
                uses *every* available step ``< final_step`` regardless of this
                filter — the influence of a sample at ``t_s`` inherently depends
                on all intervening updates — so this only selects which rows are
                reported, never which steps shape the dynamics.
            final_step: Capital ``T`` in the formula: the step index of the final
                model the test gradients were taken at.  Only training steps
                ``< final_step`` participate (both as rows and in the Fisher
                product).  ``None`` (default) uses ``max(available step) + 1``,
                i.e. every recorded training step participates.
            loss_reduction: How the *training* loss whose backward produced the
                recorded gradients was reduced over each minibatch — ``"mean"``
                (default) or ``"sum"``.  This fixes the scale of the empirical
                Fisher ``H_t`` in the propagation factor ``(I − η H_t)``.  The
                exact SGD Jacobian is ``I − η ∇²L_t``; its empirical-Fisher form
                needs the **true** per-sample gradients, ``H_t ≈ (1/c)Σ_z ĝ_zĝ_zᵀ``
                where ``ĝ_z`` is the recorded gradient and ``c`` its per-sample
                loss weight.  Under ``"mean"`` the backward scaled each ``ĝ_z`` by
                ``1/B_t`` (``c = 1/B_t``), so the Fisher is multiplied by the
                step's batch size ``B_t`` — inferred from the number of records
                at that step.  Under ``"sum"`` the recorded gradients are already
                the true per-sample gradients (``c = 1``) and no correction is
                applied.  The score's *front* factor and update direction use
                ``ĝ_z`` directly and are unaffected either way; only the Fisher
                scale changes.  (``B_t`` is taken to be the number of recorded
                samples at step ``t``, which assumes the full minibatch was
                collected.)
            verbose: Show tqdm progress bars on the logging process.
            layer_name: Restrict scoring (and the per-step Fisher) to this subset
                of the *stored* layers (``str`` or list; unknown names raise).
                ``None`` (default) uses every stored layer.  A read-time filter —
                the same cache can be re-queried per layer.
            learning_rate: The SGD learning-rate schedule ``η`` used during
                training — a float (constant) or ``{step: η_step}`` mapping
                covering every propagated step.  Enters both the per-step score
                scale and the Fisher factors ``(I − η_k H_k)``, so it must match
                the schedule the gradients were collected under (a mismatch with
                the recorded schedule warns).

        Returns:
            An :class:`AttributionScore`; also persisted to ``args.output_dir``.

        Raises:
            ValueError: If a gradients dir is missing, ``loss_reduction`` is not
                ``"mean"``/``"sum"``, ``propagation`` is not
                ``"test"``/``"train"``, ``loop_over_test`` is combined with
                ``propagation="train"``, ``selected_training_steps`` matches no
                available step, or no training step satisfies
                ``step < final_step``.
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}."
            )
        if propagation not in ("test", "train"):
            raise ValueError(
                f"propagation must be 'test' or 'train', got {propagation!r}."
            )
        if propagation == "train" and loop_over_test:
            raise ValueError(
                "loop_over_test applies only to propagation='test': the "
                "train-side sweep is test-independent (its memory is the (d, d) "
                "operator, not the test embedding), so there is nothing to "
                "block over."
            )
        if dvemb_dir is not None and propagation != "train":
            raise ValueError(
                "dvemb_dir requires propagation='train': data value embeddings "
                "only exist in the train-side sweep."
            )
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__} requires both train_gradients_dir and "
                "test_gradients_dir."
            )

        test_fm = GradientFileManager(test_gradients_dir)
        (
            train_fm, prop_steps, output_steps, final_step, learning_rate,
        ) = self._resolve_sweep(
            train_gradients_dir, selected_training_steps, final_step,
            learning_rate,
        )

        # The train source is restricted to the propagated steps; the sweep pulls
        # one step at a time from it (DiskGradientSource.for_steps).  The test
        # source supplies every column (the final-model gradients).
        layer_name = normalize_layer_names(layer_name)
        train_source = DiskGradientSource(
            train_fm, self.args, steps=prop_steps, layer_name=layer_name,
        )
        test_source = DiskGradientSource(
            test_fm, self.args, layer_name=layer_name,
            desc="DVEmb: test", verbose=verbose,
        )
        return self._run(
            train_source,
            test_source,
            prop_steps=prop_steps,
            output_steps=output_steps,
            loss_reduction=loss_reduction,
            propagation=propagation,
            dvemb_fm=GradientFileManager(dvemb_dir) if dvemb_dir else None,
            loop_over_test=loop_over_test,
            verbose=verbose,
            layer_name=layer_name,
            learning_rate=learning_rate,
            algorithm_meta={
                "final_step": final_step,
                "selected_training_steps": sorted(output_steps),
                "propagated_steps": sorted(prop_steps),
                "learning_rate": learning_rate,
                "loss_reduction": loss_reduction,
                "propagation": propagation,
                "dvemb_dir": dvemb_dir,
            },
        )

    def _collect_test_matrix(
        self, test_source: DiskGradientSource, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Materialise every test gradient into one dense per-layer embedding.

        Returns ``(w, test_ids)`` where ``w`` maps each layer to its
        ``(num_test, d_layer)`` final-model test gradients, rows ordered by
        first appearance of each test hash (duplicate-hash rows collapse via
        ``index_copy_`` — last wins).
        """
        test_ids: List[str] = []
        test_index: Dict[str, int] = {}
        pending: List[Tuple[Dict[str, torch.Tensor], List[int]]] = []
        for _step, test_g, test_hashes in test_source:
            mat = self._materialize(test_g, device)
            cols: List[int] = []
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
                cols.append(test_index[h])
            pending.append((mat, cols))
        num_test = len(test_ids)
        layers = list(pending[0][0].keys()) if pending else []
        w: Dict[str, torch.Tensor] = {
            name: torch.zeros(num_test, pending[0][0][name].shape[1], device=device)
            for name in layers
        }
        for mat, cols in pending:
            idx = torch.as_tensor(cols, device=device)
            for name in layers:
                w[name].index_copy_(0, idx, mat[name])
        return w, test_ids

    def _run(
        self,
        train_source: DiskGradientSource,
        test_source: DiskGradientSource,
        *,
        prop_steps: List[int],
        output_steps: set,
        loss_reduction: str,
        propagation: str,
        loop_over_test: bool,
        verbose: bool,
        algorithm_meta: dict,
        layer_name: Optional[List[str]] = None,
        learning_rate: Union[float, Dict[int, float]] = 1.0,
        dvemb_fm: Optional[GradientFileManager] = None,
    ) -> AttributionScore:
        """Score a train source against a test source — the shared DVEmb loop.

        With ``propagation="test"`` the test embedding ``w`` starts at the
        final-model test gradients and is back-propagated through ``prop_steps``
        (latest→earliest) by :meth:`_propagate_and_score`, pulling each step's
        train blocks from ``train_source``.  Because every test column
        propagates independently, ``loop_over_test`` trades memory for disk
        reads without changing the result.  With ``propagation="train"`` the
        same sweep instead carries the explicit ``(d, d)`` operator and embeds
        the *train* records (:meth:`_propagate_train_and_score`); the test
        matrix is held only for the final inner products.  Both sources are
        re-iterable (``reusable``); the train source is swept once per pass.
        """
        device = self.args.device
        if propagation == "train":
            # ---- train-side: embed the train records, dot against the test
            # matrix.  Layer blocks are concatenated into one flat axis so the
            # (d, d) operator carries the full (cross-layer) Fisher — exactly
            # the test-side semantics.
            w, test_ids = self._collect_test_matrix(test_source, device)
            num_test = len(test_ids)
            layers = list(w.keys())
            slices: Dict[str, Tuple[int, int]] = {}
            offset = 0
            for name in layers:
                slices[name] = (offset, offset + w[name].shape[1])
                offset = slices[name][1]
            gib = offset * offset * 4 / 2**30
            if gib > 2.0:
                warnings.warn(
                    f"propagation='train' maintains a ({offset}, {offset}) "
                    f"float32 operator (~{gib:.1f} GiB) on {device}. Project "
                    "the gradients at collection time and/or restrict "
                    "layer_name to shrink d, or use propagation='test'.",
                    stacklevel=3,
                )
            test_flat = (
                torch.cat([w[name] for name in layers], dim=1)
                if layers
                else torch.zeros(num_test, 0, device=device)
            )
            del w
            scores, row_train_ids, row_steps = self._propagate_train_and_score(
                test_flat, layers, slices, train_source, prop_steps,
                output_steps, device, loss_reduction, verbose, learning_rate,
                dvemb_fm=dvemb_fm,
            )
        elif not loop_over_test:
            # ---- step outer / test inner: one dense embedding, train read once.
            # Materialise every test gradient into one (num_test, d) embedding and
            # sweep the train gradients a single time (peak: full w + one train
            # block).  Fastest; the default.
            w, test_ids = self._collect_test_matrix(test_source, device)
            num_test = len(test_ids)
            layers = list(w.keys())
            scores, row_train_ids, row_steps = self._propagate_test_and_score(
                w, num_test, layers, train_source, prop_steps, output_steps,
                device, loss_reduction, verbose, learning_rate,
            )
        else:
            # ---- test outer: one block's embedding resident, train re-streamed.
            # Bounds peak memory to a single test block's embedding by giving each
            # test block its own full sweep (re-reading the train gradients once
            # per block).  Use when the full test embedding does not fit.
            # Pass 1 fixes the column order from hashes alone.
            test_ids = []
            test_index = {}
            for _step, _tg, test_hashes in test_source:
                for h in test_hashes:
                    if h not in test_index:
                        test_index[h] = len(test_ids)
                        test_ids.append(h)
            num_test = len(test_ids)
            # Pass 2: one test block at a time — seed w, sweep, scatter columns.
            scores = None
            row_train_ids = []
            row_steps = []
            for _step, test_g, test_hashes in test_source:
                w_block = self._materialize(test_g, device)  # (B_block, d) per layer
                layers = list(w_block.keys())
                block_cols = [test_index[h] for h in test_hashes]
                block_scores, rtids, rsteps = self._propagate_test_and_score(
                    w_block, len(block_cols), layers, train_source, prop_steps,
                    output_steps, device, loss_reduction, verbose, learning_rate,
                )
                if scores is None:
                    scores = torch.zeros(
                        block_scores.shape[0], num_test, dtype=torch.float
                    )
                    row_train_ids, row_steps = rtids, rsteps
                # Scatter this block's columns into the shared (rows × num_test).
                scores[:, block_cols] = block_scores
            if scores is None:
                scores = torch.zeros(0, num_test, dtype=torch.float)

        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta=algorithm_meta,
            algorithm=self.algorithm,
            layer_name=layer_name,
        )
        result.save(self.args.output_path)
        return result
