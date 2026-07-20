"""K-FAC / EK-FAC influence attribution over on-disk gradients (workflow 2).

Like :class:`~dattri_llm.attribution.algorithm.tracin.TracInAttributor`, these
attributors
consume :class:`~dattri_llm.gradient.gradient.Gradient` records previously
persisted by :class:`~dattri_llm.gradient.storage_manager.GradientStorageManager`; no
forward/backward pass is run at attribution time.  Unlike TracIn (a raw inner
product), they precondition the inner product by an approximate inverse Fisher
estimated *from the training gradients themselves*.

These are **single-checkpoint** methods: there is no per-step ensemble (no
``steps``/``weights``).  Every record in ``train_gradients_dir`` is one training
sample and every record in ``test_gradients_dir`` is one test sample; the score
is the full ``(num_train, num_test)`` matrix

    score[i, j] = sum_layer  vec(dW_te,j)^T F_l^-1 vec(dW_tr,i)

where the per-layer Fisher is approximated with the Kronecker structure
``F_l ~ A_l x G_l`` (``A`` the input-activation covariance, ``G`` the
output-gradient covariance) fit over the whole training set.

* **K-FAC** uses ``F_l^-1 ~ (A_l + lambda)^-1 x (G_l + lambda)^-1``.  Because the
  per-sample
  gradient factorises as ``sum_t g_t a_t^T``, this collapses to a whitened version
  of the factorised cross-gram -- no weight gradient is ever materialised.
* **EK-FAC** rotates into the Kronecker eigenbasis ``U_A, U_G`` and replaces the
  Kronecker eigenvalues with the *empirical* second moments ``Lambda`` of the
  projected gradients (a second pass over the training gradients), giving
  ``F_l^-1 ~ (U_A x U_G) (Lambda + lambda)^-1 (U_A x U_G)^T``.

Only linear and convolution layers are K-FAC-eligible; normalisation and
embedding layers (for which K-FAC is undefined) are skipped by default.  Token/
spatial positions are summed (matching a sum-over-tokens loss).  Rows and columns
of the returned :class:`~dattri_llm.attribution.score.AttributionScore` are
identified by the on-disk content hash, in disk order.

Normalisation layers are not heavily parametrised, so their per-layer Fisher can
be estimated **directly** rather than with the Kronecker factorisation.  Passing
``non_kfac_strategy="direct"`` to :meth:`attribute_from_cache` adds a dense
empirical-Fisher preconditioner ``F_l^-1`` for each such layer (built from the
token-summed ``(B, d)`` weight gradients), whose contribution is summed into the
K-FAC score.  Layers stored **materialized** -- e.g. a TRAK-projected
(``factorize=False``) capture, a dense ``(B, proj_dim)`` tensor with no
``(a, g)`` factors -- can never enter K-FAC; they are **always** preconditioned
by the direct dense Fisher (with a warning), regardless of
``non_kfac_strategy``, which governs the norm layers only.  Layers whose
parameter count exceeds ``direct_fim_max_params`` are left out to bound the
``O(d^2)`` Fisher; factorized embedding layers (heavily parametrised) stay
ignored.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from dattri_llm.attribution.base import BaseAttributor
from dattri_llm.attribution.score import AttributionScore
from dattri_llm.attribution.utils import (
    collect_to_disk,
    normalize_layer_names,
    score_sources,
    task_loss_fn,
)
from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import (
    DiskGradientSource,
    GradientSource,
    GradientStreamer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.hooks import HookManagerConfig


class _KroneckerBaseAttributor(BaseAttributor):
    """Shared on-disk plumbing for the K-FAC family.

    Subclasses implement :meth:`_fit` (estimate the per-layer preconditioner from
    the whole training set), :meth:`_prepare_test` (build a test block's scoring
    representation), and :meth:`_score` (preconditioned cross-gram for a train
    block vs a prepared test rep).  Everything else -- file-granular block
    iteration, ``loop_over_test`` memory control, column bookkeeping, and the
    ``(num_train, num_test)`` assembly -- is shared.
    """

    algorithm: str = "Kronecker"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
    ) -> None:
        self.args = args
        self.task = task
        # Per-run bookkeeping, reset by _run(): embedding layers the direct
        # Fisher left uncovered, K-FAC-typed layers diverted to the dense
        # Fisher because they were stored materialized (no factors to build
        # covariances from), and whether norm layers enter the Fisher too
        # (the ``non_kfac_strategy="direct"`` choice).
        self._fisher_saw_embedding: set[str] = set()
        self._skipped_materialized: set[str] = set()
        self._direct_norm_layers: bool = False

    def cache(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        cache_dir: str | None = None,
        hook_config: HookManagerConfig | None = None,
    ) -> list[tuple[str, str]]:
        """Collect the gradients K-FAC/EK-FAC needs, live, to disk.

        Both sides are frozen probes at the task's **first** checkpoint (K-FAC and
        EK-FAC are single-checkpoint); the train gradients also build the Fisher.
        Reproducing :meth:`attribute` is *cache + attribute_from_cache* on the
        single returned pair.  A list (of length 1) is returned for parity with
        the multi-checkpoint :class:`TracInAttributor.cache`.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            cache_dir: Parent dir; the pair goes under ``<cache_dir>/ckpt_0/``.
                Defaults to ``args.output_dir``.
            hook_config: :class:`HookManagerConfig` for the internal streamers
                (which layers to hook, per-layer projection, ...).  ``None`` uses
                the streamer default (factorized hooks on every linear-family layer).

        Returns:
            A one-element list ``[(train_gradients_dir, test_gradients_dir)]``.
        """
        if self.task is None:
            raise ValueError(
                "cache() (live collection) requires a ``task`` with a model; pass "
                "pre-collected gradients to attribute_from_cache() instead.",
            )
        n_ckpt = len(self.task.get_checkpoints())
        if n_ckpt > 1:
            warnings.warn(
                f"{type(self).__name__} is single-checkpoint; only checkpoint 0 is "
                f"used and the other {n_ckpt - 1} provided checkpoint(s) are ignored.",
                stacklevel=2,
            )
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        train_dir = str(Path(cache_dir) / "ckpt_0" / "train_grads")
        test_dir = str(Path(cache_dir) / "ckpt_0" / "test_grads")

        self.task._load_checkpoints(0)
        model = self.task.get_model()
        collect_to_disk(
            GradientStreamer(
                model,
                train_dataset,
                self.args,
                batch_size=self.args.per_device_train_batch_size,
                enable_update=False,
                loss_fn=task_loss_fn(self.task.original_loss_func),
                config=hook_config,
            ),
            GradientStorageManager(train_dir),
        )
        collect_to_disk(
            GradientStreamer(
                model,
                test_dataset,
                self.args,
                batch_size=self.args.per_device_eval_batch_size,
                enable_update=False,
                loss_fn=task_loss_fn(self.task.original_target_func),
                config=hook_config,
            ),
            GradientStorageManager(test_dir),
        )
        return [(train_dir, test_dir)]

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _fit(
        self,
        train_source: GradientSource,
        device: torch.device,
        fisher_acc: ops.FisherAccumulator,
        damping: float,
    ) -> object:
        """Estimate the per-layer K-FAC preconditioner from the training gradients.

        Iterates ``train_source`` (a re-iterable ``GradientSource`` -- disk or a
        frozen streamer; EK-FAC iterates it twice).  Returns an opaque context
        object (possibly empty if no K-FAC-eligible layer is present) passed back
        to :meth:`_prepare_test` and :meth:`_score`.

        The implementation must also feed every block to *fisher_acc* (via
        :meth:`_accumulate_fisher`) during its **first** pass, so the
        direct-Fisher estimate for the non-K-FAC layers (materialized layers
        always; norm layers under ``non_kfac_strategy="direct"``) reuses that
        sweep.
        """

    @abstractmethod
    def _prepare_test(self, test_g: Gradient, ctx: object) -> object:
        """Build the per-block test representation used for scoring.

        Split out from :meth:`_score` so it can be computed once and reused
        across all train blocks (``loop_over_test=False``), or recomputed and
        discarded per train block (``loop_over_test=True``) to bound memory.
        """

    @abstractmethod
    def _score(
        self,
        train_g: Gradient,
        test_rep: object,
        ctx: object,
    ) -> torch.Tensor:
        """``(B_train, B_test)`` preconditioned score for a train block against a
        test representation from :meth:`_prepare_test`.
        """

    # ------------------------------------------------------------------ #
    # Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def _fit_context(
        self,
        train_source: GradientSource,
        device: torch.device,
        damping: float,
        non_kfac_strategy: str,
        direct_fim_max_params: int,
    ) -> tuple[dict, dict[str, torch.Tensor]]:
        """Fit the K-FAC preconditioner and the direct empirical Fisher.

        One (or, for EK-FAC, two) sweep(s) over ``train_source``.  The
        empirical-Fisher accumulator is filled in the *same* first pass: it
        always receives layers stored materialized (K-FAC is impossible for
        them, so the dense Fisher is their only preconditioner), and
        additionally the norm layers when the direct strategy is requested.
        Returns ``(ctx, fim_ctx)``.
        """
        self._fisher_saw_embedding = set()
        self._skipped_materialized = set()
        self._direct_norm_layers = non_kfac_strategy == "direct"
        fisher_acc = ops.FisherAccumulator(direct_fim_max_params)
        ctx = self._fit(train_source, device, fisher_acc, damping)
        if self._skipped_materialized:
            warnings.warn(
                "Layers stored materialized (e.g. a TRAK-projected capture) "
                "cannot enter K-FAC -- there are no factorized (a, g) factors "
                "to build the Kronecker covariances from.  Preconditioning "
                "them with the direct dense empirical Fisher (FIM) instead, "
                f"bounded by direct_fim_max_params={direct_fim_max_params}: "
                f"{sorted(self._skipped_materialized)}.  Collect with "
                "factorize=True (LoGRA) to keep them K-FAC-eligible.",
                stacklevel=2,
            )
        fim_ctx: dict[str, torch.Tensor] = self._finalize_fisher(
            fisher_acc,
            direct_fim_max_params,
            damping,
        )
        if not ctx and not fim_ctx:
            raise ValueError(
                "No eligible layers found in the training gradients: no "
                "K-FAC-eligible (linear/conv) layer"
                + (
                    " and no direct-Fisher (norm) layer within "
                    f"direct_fim_max_params={direct_fim_max_params}"
                    if non_kfac_strategy == "direct"
                    else " (pass non_kfac_strategy='direct' to include norm layers)"
                )
                + ". Check the hook config and the collected layers.",
            )
        return ctx, fim_ctx

    def _write_preconditioned_store(
        self,
        test_source: GradientSource,
        ctx: dict,
        fim_ctx: dict[str, torch.Tensor],
        out_dir: str,
        device: torch.device,
    ) -> str:
        """Persist fully preconditioned test representations to *out_dir*.

        One sweep over ``test_source``: every block's K-FAC/EK-FAC and
        direct-Fisher representations are computed via :meth:`_prepare_test` /
        :meth:`_prepare_fim_test` (which carry the entire preconditioner on
        the test side), **materialized** to one ``(B_te, D)`` tensor per
        layer, and stored as materialized :class:`Gradient` records with the
        source records' steps and hashes.  Scoring against the store is a
        plain inner product, so it composes with
        :class:`~dattri_llm.attribution.algorithm.tracin.TracInAttributor.attribute_from_cache`
        exactly like a DVEmb embedding store.
        """
        fm = GradientStorageManager(str(out_dir))
        for step, test_block, hashes in test_source:
            test_g = test_block.to(device)
            kfac_rep = self._prepare_test(test_g, ctx)
            fim_rep = self._prepare_fim_test(test_g, fim_ctx)
            data: dict[str, torch.Tensor] = {}
            layer_types: dict[str, str] = {}
            for layer, entry in kfac_rep.items():
                if isinstance(entry, tuple):
                    # K-FAC whitened factors -> materialize for storage.
                    a_w, g_w = entry
                    tensor = ops._materialize(
                        a_w,
                        g_w,
                        test_g.layer_types[layer],
                    )
                else:  # already a (B_te, D) preconditioned matrix (EK-FAC)
                    tensor = entry
                data[layer] = tensor.detach().cpu()
                layer_types[layer] = test_g.layer_types[layer]
            for layer, tensor in fim_rep.items():
                data[layer] = tensor.detach().cpu()
                layer_types[layer] = test_g.layer_types[layer]
            grad = Gradient(
                representation=dict.fromkeys(data, "materialized"),
                data=data,
                layer_types=layer_types,
            )
            fm.save_bulk(
                [GradientRecord(step=step, input_hash=list(hashes), gradient=grad)],
            )
        return str(out_dir)

    def _run(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
        *,
        damping: float = 1e-3,
        loop_over_test: bool = False,
        preconditioned_test_dir: str | None = None,
        non_kfac_strategy: Literal["ignore", "direct"] = "ignore",
        direct_fim_max_params: int = 4096,
        algorithm_meta_extra: dict | None = None,
        layer_name: list[str] | None = None,
    ) -> AttributionScore:
        """Fit the K-FAC preconditioner from ``train_source``, then score it
        against ``test_source`` -- the shared loop behind :meth:`attribute_from_cache`
        and :meth:`attribute`.

        ``train_source`` must be **re-iterable**: the Fisher pre-pass re-reads the
        train gradients before scoring (EK-FAC reads them twice more), so a
        single-shot trajectory stream is rejected.

        With ``loop_over_test=True`` and *preconditioned_test_dir* set, the
        preconditioned test representations are computed once, persisted
        there, and re-streamed **from disk** on every per-train-block sweep
        (instead of being recomputed from the raw test gradients each time).
        """
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}.")
        if preconditioned_test_dir is not None and not loop_over_test:
            raise ValueError(
                "preconditioned_test_dir only applies to loop_over_test=True "
                "(with loop_over_test=False the preconditioned representations "
                "are simply held in memory).",
            )
        if non_kfac_strategy not in ("ignore", "direct"):
            raise ValueError(
                "non_kfac_strategy must be 'ignore' or 'direct', got "
                f"{non_kfac_strategy!r}.",
            )
        if direct_fim_max_params <= 0:
            raise ValueError(
                f"direct_fim_max_params must be positive, got {direct_fim_max_params}.",
            )
        if not getattr(train_source, "reusable", False):
            raise ValueError(
                f"{type(self).__name__} needs a re-iterable train source: the "
                "Fisher pre-pass re-reads the train gradients before scoring. Use "
                "on-disk gradients or a frozen GradientStreamer (enable_update=False, "
                "single-shot trajectory stream).",
            )

        device = self.args.device
        ctx, fim_ctx = self._fit_context(
            train_source,
            device,
            damping,
            non_kfac_strategy,
            direct_fim_max_params,
        )

        # Per test block: (K-FAC rep, direct-Fisher rep).  Per (train, test) pair:
        # the summed K-FAC + direct-Fisher score.
        def prepare(test_g: Gradient) -> tuple[object, dict[str, torch.Tensor]]:
            return self._prepare_test(test_g, ctx), self._prepare_fim_test(
                test_g,
                fim_ctx,
            )

        if preconditioned_test_dir is not None:
            # Precondition once, persist, and re-stream the (small) store per
            # train block; records already carry the full preconditioner, so
            # "preparing" a disk block is just splitting its layers.
            self._write_preconditioned_store(
                test_source,
                ctx,
                fim_ctx,
                preconditioned_test_dir,
                device,
            )
            test_source = DiskGradientSource(
                GradientStorageManager(preconditioned_test_dir),
                self.args,
                desc=f"{self.algorithm}: preconditioned test",
            )

            def prepare(  # noqa: F811 -- intentional swap of the prepare hook
                test_g: Gradient,
            ) -> tuple[object, dict[str, torch.Tensor]]:
                kfac_rep = {
                    layer: tensor
                    for layer, tensor in test_g.data.items()
                    if layer in ctx
                }
                fim_rep = {
                    layer: tensor.float()
                    for layer, tensor in test_g.data.items()
                    if layer in fim_ctx
                }
                return kfac_rep, fim_rep

        def score(train_g: Gradient, rep: object, n_test: int) -> torch.Tensor:
            test_rep, fim_rep = rep
            return self._combine_score(train_g, test_rep, fim_rep, ctx, fim_ctx, n_test)

        scores, row_train_ids, row_steps, test_ids = score_sources(
            train_source,
            test_source,
            device,
            prepare_test=prepare,
            score_block=score,
            loop_over_test=loop_over_test,
        )

        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            # One row per train sample, stamped with the step it was recorded at.
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta={
                "damping": damping,
                "non_kfac_strategy": non_kfac_strategy,
                "direct_fim_layers": sorted(fim_ctx),
                **(algorithm_meta_extra or {}),
            },
            algorithm=self.algorithm,
            layer_name=layer_name,
        )
        result.save(self.args.output_path)
        return result

    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        *,
        damping: float = 1e-3,
        selected_training_steps: Iterable[int] | None = None,
        loop_over_test: bool = False,
        preconditioned_test_dir: str | None = None,
        verbose: bool = False,
        non_kfac_strategy: Literal["ignore", "direct"] = "ignore",
        direct_fim_max_params: int = 4096,
        layer_name: str | list[str] | None = None,
    ) -> AttributionScore:
        """Score pre-collected on-disk gradients (the *store-then-attribute* path).

        The Fisher is estimated from the (selected) train gradients; the test set
        supplies every column.

        Args:
            train_gradients_dir: Directory written by
                :class:`GradientStorageManager` for the train pass.
            test_gradients_dir: Directory written by
                :class:`GradientStorageManager` for the test pass.
            damping: Tikhonov term added to each covariance factor (K-FAC) or
                to the corrected eigenvalues (EK-FAC) before inversion.
            selected_training_steps: Restrict the train checkpoints (Fisher fit +
                output rows) to these steps; ``None`` uses all on disk.
            loop_over_test: Re-stream + rebuild the test reps per train block (low
                memory) instead of caching them once (default).
            preconditioned_test_dir: With ``loop_over_test=True``, persist the
                preconditioned test representations here and re-stream them
                from disk on every sweep, instead of recomputing them from
                the raw test gradients each time.  See
                :meth:`cache_preconditioned_test` to build this store ahead
                of time (e.g. once, for reuse across many
                ``attribute_from_cache`` calls with the same test set).
            verbose: Show tqdm progress bars on the logging process.
            non_kfac_strategy: Governs the **norm** layers (K-FAC is
                undefined for them): ``"ignore"`` (default) skips them;
                ``"direct"`` adds a dense empirical-Fisher preconditioner for
                each, bounded by ``direct_fim_max_params``.  Layers stored
                materialized (e.g. a TRAK-projected capture) cannot enter
                K-FAC and always take the dense-Fisher path (with a
                warning), under either strategy.
            direct_fim_max_params: Parameter-count cap for the dense
                per-layer Fisher under ``non_kfac_strategy="direct"``.
            layer_name: Restrict scoring (and the Fisher fit) to this subset of the
                *stored* layers (``str`` or list; unknown names raise).  ``None``
                (default) uses every stored layer.  A read-time filter -- the same
                cache can be re-queried per layer.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__} requires both train_gradients_dir and "
                "test_gradients_dir.",
            )
        layer_name = normalize_layer_names(layer_name)
        train_fm = GradientStorageManager(train_gradients_dir)
        test_fm = GradientStorageManager(test_gradients_dir)
        train = DiskGradientSource(
            train_fm,
            self.args,
            steps=selected_training_steps,
            layer_name=layer_name,
            desc=f"{self.algorithm}: train",
            verbose=verbose,
        )
        test = DiskGradientSource(
            test_fm,
            self.args,
            layer_name=layer_name,
            desc=f"{self.algorithm}: preparing test",
            verbose=verbose,
        )
        return self._run(
            train,
            test,
            damping=damping,
            loop_over_test=loop_over_test,
            preconditioned_test_dir=preconditioned_test_dir,
            non_kfac_strategy=non_kfac_strategy,
            direct_fim_max_params=direct_fim_max_params,
            algorithm_meta_extra={
                "selected_training_steps": train._steps,
                "sample_id_key": {
                    "train": train_fm.sample_id_key,
                    "test": test_fm.sample_id_key,
                },
            },
            layer_name=layer_name,
        )

    def cache_preconditioned_test(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        preconditioned_test_dir: str | None = None,
        *,
        damping: float = 1e-3,
        selected_training_steps: Iterable[int] | None = None,
        non_kfac_strategy: Literal["ignore", "direct"] = "ignore",
        direct_fim_max_params: int = 4096,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """Fit the preconditioner and persist **preconditioned** test reps.

        Mirrors ``DVEmbAttributor.cache_dvemb``: a one-time sweep that turns
        raw test gradients into a store already carrying the full K-FAC/EK-FAC
        (and direct-Fisher) preconditioner applied on the test side (see the
        module docstring and :func:`dattri_llm.gradient.ops.kfac_precondition`
        / :func:`~dattri_llm.gradient.ops.ekfac_precondition` for the identity
        this relies on).  Scoring the store against train gradients then
        reduces to a plain ``TracInAttributor`` inner product -- no
        preconditioner is recomputed::

            dvemb_dir = attr.cache_preconditioned_test(train_dir, test_dir)
            scores = TracInAttributor(args).attribute_from_cache(
                train_gradients_dir=train_dir, test_gradients_dir=dvemb_dir)

        and this store can be reused across any later call, or passed via
        ``attribute_from_cache(..., loop_over_test=True,
        preconditioned_test_dir=...)`` to persist it inline instead of
        pre-building it.

        Args:
            train_gradients_dir: Directory written by :class:`GradientStorageManager`
                for the train pass (fits the preconditioner).
            test_gradients_dir: Directory written for the test pass (the raw
                gradients that get preconditioned).
            preconditioned_test_dir: Where to store the result; defaults to
                ``<args.output_dir>/<algorithm>_preconditioned_test``.
            damping: As in :meth:`attribute_from_cache`.
            selected_training_steps: As in :meth:`attribute_from_cache`
                (restricts the Fisher fit, not what gets stored).
            non_kfac_strategy: As in :meth:`attribute_from_cache`.
            direct_fim_max_params: As in :meth:`attribute_from_cache`.
            layer_name: As in :meth:`attribute_from_cache`.
            verbose: Show tqdm progress bars on the logging process.

        Returns:
            ``preconditioned_test_dir``.
        """
        if preconditioned_test_dir is None:
            subdir = f"{self.algorithm.lower()}_preconditioned_test"
            preconditioned_test_dir = str(Path(self.args.output_dir) / subdir)
        layer_name = normalize_layer_names(layer_name)
        train = DiskGradientSource(
            GradientStorageManager(train_gradients_dir),
            self.args,
            steps=selected_training_steps,
            layer_name=layer_name,
            desc=f"{self.algorithm}: train (fitting)",
            verbose=verbose,
        )
        test = DiskGradientSource(
            GradientStorageManager(test_gradients_dir),
            self.args,
            layer_name=layer_name,
            desc=f"{self.algorithm}: test (raw)",
            verbose=verbose,
        )
        if not getattr(train, "reusable", False):
            raise ValueError(
                f"{type(self).__name__} needs a re-iterable train source to fit "
                "the preconditioner; use on-disk gradients.",
            )
        device = self.args.device
        ctx, fim_ctx = self._fit_context(
            train,
            device,
            damping,
            non_kfac_strategy,
            direct_fim_max_params,
        )
        return self._write_preconditioned_store(
            test,
            ctx,
            fim_ctx,
            preconditioned_test_dir,
            device,
        )

    def attribute(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        damping: float = 1e-3,
        loop_over_test: bool = False,
        non_kfac_strategy: Literal["ignore", "direct"] = "ignore",
        direct_fim_max_params: int = 4096,
        hook_config: HookManagerConfig | None = None,
    ) -> AttributionScore:
        """Score by collecting gradients **live** at the task's first checkpoint.

        K-FAC/EK-FAC are **single-checkpoint** methods, so only the first
        checkpoint of the task is used (matching dattri).  Both sides are frozen
        probes; the model, loss (and optional ``target_func`` for the test side),
        and the batch sizes come from the task and ``args``.  ``hook_config``
        configures the internal streamers' capture (which layers to hook,
        per-layer projection, ...); ``None`` uses the streamer default.  The test
        streamer shares the train streamer's hooks, so one config governs both.
        """
        if self.task is None:
            raise ValueError(
                "attribute() (live collection) requires a ``task`` with a model; "
                "use attribute_from_cache().",
            )
        n_ckpt = len(self.task.get_checkpoints())
        if n_ckpt > 1:
            warnings.warn(
                f"{type(self).__name__} is single-checkpoint; only checkpoint 0 is "
                f"used and the other {n_ckpt - 1} provided checkpoint(s) are "
                "ignored.",
                stacklevel=2,
            )
        self.task._load_checkpoints(0)
        model = self.task.get_model()
        train = GradientStreamer(
            model,
            train_dataset,
            self.args,
            batch_size=self.args.per_device_train_batch_size,
            loss_fn=task_loss_fn(self.task.original_loss_func),
            config=hook_config,
        )
        test = GradientStreamer(
            model,
            test_dataset,
            self.args,
            batch_size=self.args.per_device_eval_batch_size,
            loss_fn=task_loss_fn(self.task.original_target_func),
            hook_manager=train.hook_manager,  # one set of hooks over the model
        )
        with train, test:
            return self._run(
                train,
                test,
                damping=damping,
                loop_over_test=loop_over_test,
                non_kfac_strategy=non_kfac_strategy,
                direct_fim_max_params=direct_fim_max_params,
                layer_name=train.hook_manager.layer_name,  # what was hooked
            )

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _kfac_layers(self, grad: Gradient) -> list[str]:
        """Layer names eligible for K-FAC: linear/conv **stored factorized**.

        A layer of eligible type stored materialized -- e.g. a TRAK-projected
        (``factorize=False``) capture, which keeps its original layer type
        but holds a dense ``(B, proj_dim)`` tensor -- has no ``(a, g)``
        factors to build the Kronecker covariances from.  Such layers are
        recorded in ``_skipped_materialized`` (warned about once per run) and
        left to the direct-Fisher fallback.
        """
        names = []
        for name, value in grad.data.items():
            lt = grad.layer_types[name]
            if not (ops.is_linear(lt) or ops.is_conv(lt) or ops.is_conv_transpose(lt)):
                continue
            if isinstance(value, Factorized):
                names.append(name)
            else:
                self._skipped_materialized.add(name)
        return names

    def _fisher_layers(self, grad: Gradient) -> list[str]:
        """Layer names entering the dense empirical Fisher (FIM).

        Layers stored **materialized** (e.g. a TRAK-projected capture) enter
        **unconditionally**: K-FAC is impossible without the ``(a, g)``
        factors, so the dense Fisher over their per-sample ``(B, P)`` rows is
        their only preconditioner.  Norm layers (K-FAC undefined) enter only
        under ``non_kfac_strategy="direct"``.  Batch-level ``param_grad``
        tensors carry no per-sample axis and are excluded.
        """
        names = []
        for name, value in grad.data.items():
            lt = grad.layer_types[name]
            if lt == ops.PARAM_GRAD_TYPES:
                continue
            if isinstance(value, torch.Tensor) or (
                self._direct_norm_layers and ops.is_norm(lt)
            ):
                names.append(name)
        return names

    # ------------------------------------------------------------------ #
    # Direct-Fisher fallback for non-K-FAC (norm) layers                  #
    # ------------------------------------------------------------------ #

    def _accumulate_fisher(
        self,
        fisher_acc: ops.FisherAccumulator,
        grad: Gradient,
    ) -> None:
        """Fold one streamed training block into the per-layer Fisher estimate.

        Called from each subclass's :meth:`_fit` first pass.  Also records any
        embedding layers seen so :meth:`_finalize_fisher` can warn that they were
        left ignored (heavily parametrised -- not covered by the direct fallback).
        """
        fisher_acc.update(grad, self._fisher_layers(grad))
        # Materialized (e.g. TRAK-projected) embeddings are dense (B, P) rows
        # and enter the Fisher like any other layer; only factorized
        # embeddings stay uncovered (their materialized width is the full
        # vocab -- far past any sensible max_params).  The warning is only
        # meaningful under the direct strategy, which advertises non-K-FAC
        # coverage.
        if self._direct_norm_layers:
            self._fisher_saw_embedding.update(
                name
                for name, value in grad.data.items()
                if ops.is_embedding(grad.layer_types[name])
                and isinstance(value, Factorized)
            )

    def _finalize_fisher(
        self,
        fisher_acc: ops.FisherAccumulator,
        max_params: int,
        damping: float,
    ) -> dict[str, torch.Tensor]:
        """Turn the accumulated Fishers into ``{layer: F_l^-1}``, warning about
        the layers dropped by the ``max_params`` cap and the ignored embeddings.
        """
        if fisher_acc.skipped:
            warnings.warn(
                "The direct dense-Fisher fallback skipped layers whose "
                f"parameter count exceeds direct_fim_max_params={max_params}: "
                f"{dict(sorted(fisher_acc.skipped.items()))}.",
                stacklevel=2,
            )
        if self._fisher_saw_embedding:
            warnings.warn(
                "non_kfac_strategy='direct' does not cover embedding layers "
                f"(heavily parametrised); leaving ignored: "
                f"{sorted(self._fisher_saw_embedding)}.",
                stacklevel=2,
            )
        return {
            layer: ops.sym_inverse(F, damping)
            for layer, F in fisher_acc.result().items()
        }

    @staticmethod
    def _fisher_grad(grad: Gradient, layer: str) -> torch.Tensor:
        """Per-sample ``(B, P)`` weight gradient used for direct-Fisher scoring.

        A layer stored materialized (e.g. TRAK-projected) already *is* the
        dense per-sample representation and is used as-is.
        """
        value = grad.data[layer]
        if isinstance(value, torch.Tensor):
            return value.reshape(value.shape[0], -1).float()
        return ops.materialize(value, grad.layer_types[layer])

    def _prepare_fim_test(
        self,
        test_g: Gradient,
        fim_ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``(B_te, P)`` **preconditioned** weight gradient per direct-Fisher layer.

        ``F_l^-1`` is symmetric, so it is applied wholly on this (small) side,
        once -- each train block then needs only a plain dot.
        """
        return {
            layer: self._fisher_grad(test_g, layer).float() @ fim_ctx[layer]
            for layer in fim_ctx
            if layer in test_g.data
        }

    def _score_fim(
        self,
        train_g: Gradient,
        fim_test_rep: dict[str, torch.Tensor],
        fim_ctx: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """``(B_tr, B_te)`` direct-Fisher score, or ``None`` if no layer applies.

        ``fim_test_rep`` already carries ``F^-1``, so this is a raw dot.
        """
        total: torch.Tensor | None = None
        for layer in fim_ctx:
            if layer not in train_g.data or layer not in fim_test_rep:
                continue
            M_tr = self._fisher_grad(train_g, layer).float()  # (B_tr, P)
            R_te = fim_test_rep[layer]  # (B_te, P), preconditioned
            block = M_tr @ R_te.T  # (B_tr, B_te)
            total = block if total is None else total + block
        return total

    def _combine_score(
        self,
        train_g: Gradient,
        test_rep: object,
        fim_test_rep: dict[str, torch.Tensor],
        ctx: object,
        fim_ctx: dict[str, torch.Tensor],
        n_test: int,
    ) -> torch.Tensor:
        """Sum the K-FAC and direct-Fisher contributions for one train/test pair.

        Either side may be empty (no eligible layers); the other carries the
        score.  At least one is non-empty (enforced by ``attribute_from_cache``);
        *n_test* is the test block's column count, used only when neither side
        shares a layer with this block (an all-zero contribution).
        """
        block: torch.Tensor | None = None
        if ctx:
            block = self._score(train_g, test_rep, ctx)
        fim_block = self._score_fim(train_g, fim_test_rep, fim_ctx)
        if fim_block is not None:
            block = fim_block if block is None else block + fim_block
        if block is None:
            block = torch.zeros(train_g.batch_size, n_test)
        return block


class KFACAttributor(_KroneckerBaseAttributor):
    """K-FAC influence attributor.

    ``F_l^-1 ~ (A_l + lambda)^-1 x (G_l + lambda)^-1`` per linear/conv layer, with
    ``lambda`` the
    ``damping`` term.  See the module docstring for the score definition.

    Args:
        args: :class:`AttributionArguments` (``dataloader_*``, ``device``,
            ``output_dir`` are consulted).
        task: Accepted for API parity; unused.

    The ``damping`` term is a per-attribution argument of :meth:`attribute` and
    :meth:`attribute_from_cache`.

    The training checkpoints used are chosen per call via
    :meth:`attribute`'s ``selected_training_steps`` argument.
    """

    algorithm = "KFAC"

    def _fit(
        self,
        train_source: GradientSource,
        device: torch.device,
        fisher_acc: ops.FisherAccumulator,
        damping: float,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            # Reuse this single sweep to fit the direct Fisher (materialized
            # layers always; norm layers under the direct strategy).
            self._accumulate_fisher(fisher_acc, train_g)
        return {
            layer: (
                ops.sym_inverse(A, damping),
                ops.sym_inverse(G, damping),
            )
            for layer, (A, G) in kron.result().items()
        }

    @staticmethod
    def _prepare_test(
        test_g: Gradient,
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, object]:
        # Whiten the *test* factors by the inverse covariances **once**.  The
        # inverses are symmetric, so the preconditioner can live entirely on
        # this (small) side; each train block then enters the cross-gram raw --
        # this is the score-pass hotspot removed by P0 (train-side whitening
        # used to run once per train block).
        rep: dict[str, object] = {}
        for layer, (A_inv, G_inv) in ctx.items():
            if layer not in test_g.data:
                continue
            rep[layer] = ops.kfac_precondition(
                test_g.data[layer],
                test_g.layer_types[layer],
                A_inv,
                G_inv,
            )
        return rep

    @staticmethod
    def _score(
        train_g: Gradient,
        test_rep: dict[str, object],
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor | None:
        total: torch.Tensor | None = None
        for layer in ctx:
            rep = test_rep.get(layer)
            if rep is None or layer not in train_g.data:
                continue
            if isinstance(rep, torch.Tensor):
                # Preconditioned-materialized rep (e.g. read back from a
                # ``cache_preconditioned_test`` store): plain dot against the
                # raw materialized train gradient.
                M_tr = ops.materialize(
                    train_g.data[layer],
                    train_g.layer_types[layer],
                )
                block = M_tr @ rep.to(M_tr.dtype).T
            else:
                # Whitened-factor rep: raw train factors cross-grammed against
                # the (test-side) whitened factors -- equal to kfac_cross by
                # symmetry of the inverses, without per-block whitening.
                a1, g1 = ops.preprocess_factorized(
                    train_g.data[layer],
                    train_g.layer_types[layer],
                )
                block = ops._cross_gram(a1, g1, *rep, train_g.layer_types[layer])
            total = block if total is None else total + block
        return total  # None when no layer overlaps; _combine_score zero-fills


class EKFACAttributor(_KroneckerBaseAttributor):
    """EK-FAC influence attributor.

    Rotates each layer's gradients into the Kronecker eigenbasis ``(U_A, U_G)``
    and replaces the Kronecker eigenvalues with the empirical second moments
    ``Lambda`` of the projected gradients (a second pass over the training
    gradients), giving ``F_l^-1 ~ (U_A x U_G)(Lambda + lambda)^-1(U_A x U_G)^T``.

    The per-sample gradient is projected as ``M = U_G^T dW U_A`` -- the faithful
    expansion of ``(U_A x U_G)^T vec(dW)``.  This is the unique projection that
    **reduces to K-FAC** when ``Lambda`` equals the Kronecker eigenvalues, and it is
    invariant to the (arbitrary) sign of each eigenvector.

    ``mode`` selects the implementation and is kept mainly for backward
    compatibility / cross-checking:

    * ``"exact"`` *(default)* -- the faithful projection above.
    * ``"approx"`` -- the code path mirroring the ``dattri`` library.  Its
      original transposed projection ``U_G dW U_A^T`` was wrong (does not reduce
      to K-FAC and is sign-sensitive; see
      ``test_transposed_projection_is_sign_sensitive``); fixed, it now uses the
      same faithful projection, so the two modes produce identical scores.

    Args:
        args: :class:`AttributionArguments`.
        task: Accepted for API parity; unused.
        mode: ``"exact"`` (default) or ``"approx"``; see above.  Currently
            equivalent.

    The ``damping`` term is a per-attribution argument of :meth:`attribute` and
    :meth:`attribute_from_cache`.

    The training checkpoints used are chosen per call via
    :meth:`attribute`'s ``selected_training_steps`` argument.
    """

    algorithm = "EKFAC"
    EKFAC_MODES = ("exact", "approx")

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
        mode: str = "exact",
    ) -> None:
        if mode not in self.EKFAC_MODES:
            raise ValueError(
                f"mode must be one of {self.EKFAC_MODES}, got {mode!r}.",
            )
        super().__init__(args, task=task)
        self.mode = mode

    def _fit(
        self,
        train_source: GradientSource,
        device: torch.device,
        fisher_acc: ops.FisherAccumulator,
        damping: float,
    ) -> dict:
        # Pass 1 -- Kronecker covariance factors and their eigenbases (and the
        # direct Fisher -- materialized layers always, norm layers under the
        # direct strategy -- from the same sweep).
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            self._accumulate_fisher(fisher_acc, train_g)
        # Eigenvectors are fed to ``ekfac_materialize`` (which does ``a @ U``),
        # giving the faithful projection ``M = U_G^T dW U_A``.
        eig: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, (A, G) in kron.result().items():
            _, U_A, _, U_G = ops.kfac_eigh(A, G)
            eig[layer] = (U_A, U_G)

        # Pass 2 -- empirical second moments of the projected gradients (Lambda).
        # Skipped entirely when no K-FAC layer is present (norm-only + direct),
        # so the data is not streamed for nothing.
        lam_sum: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        for _step, train_block, _ in train_source if eig else ():
            train_g = train_block.to(device)
            for layer, (U_A, U_G) in eig.items():
                if layer not in train_g.data:
                    continue
                M = ops.ekfac_materialize(
                    train_g.data[layer],
                    train_g.layer_types[layer],
                    U_A,
                    U_G,
                )  # (B, D)
                lam_sum[layer] = lam_sum.get(layer, 0) + (M * M).sum(0)
                counts[layer] = counts.get(layer, 0) + M.shape[0]

        # Damping is folded into the stored eigenvalues here, at fit time, so
        # scoring divides by the damped spectrum directly.
        layers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer, (U_A, U_G) in eig.items():
            layers[layer] = (U_A, U_G, lam_sum[layer] / counts[layer] + damping)
        return layers

    @staticmethod
    def _prepare_test(
        test_g: Gradient,
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # Apply the *entire* damped EK-FAC inverse to the test gradients once:
        # rotate into the eigenbasis, divide by the corrected spectrum, and
        # rotate **back out** (``R = U_G (M/lam) U_A^T``).  Scoring then dots
        # raw train gradients against ``R`` -- the per-train-block eigenbasis
        # rotations (the score-pass hotspot removed by P0) disappear.
        rep: dict[str, torch.Tensor] = {}
        for layer, (U_A, U_G, lam) in ctx.items():
            if layer not in test_g.data:
                continue
            M = ops.ekfac_materialize(
                test_g.data[layer],
                test_g.layer_types[layer],
                U_A,
                U_G,
            )  # (B_te, D) eigenbasis coordinates
            rep[layer] = ops.ekfac_precondition(
                M,
                U_A,
                U_G,
                lam,
                test_g.layer_types[layer],
            )
        return rep

    @staticmethod
    def _score(
        train_g: Gradient,
        test_mats: dict[str, torch.Tensor],
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor | None:
        total: torch.Tensor | None = None
        for layer in ctx:
            M_te = test_mats.get(layer)  # (B_te, D), fully preconditioned
            if M_te is None or layer not in train_g.data:
                continue
            M_tr = ops.materialize(
                train_g.data[layer],
                train_g.layer_types[layer],
            )  # (B_tr, D) raw weight gradient -- no rotation
            block = M_tr @ M_te.to(M_tr.dtype).T  # (B_tr, B_te)
            total = block if total is None else total + block
        return total  # None when no layer overlaps; _combine_score zero-fills
