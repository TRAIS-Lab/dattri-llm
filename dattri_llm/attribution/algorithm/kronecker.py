"""K-FAC / EK-FAC influence attribution over on-disk gradients (workflow 2).

Like :class:`~dattri_llm.attribution.algorithm.tracin.TracInAttributor`, these
attributors
consume :class:`~dattri_llm.gradient.gradient.Gradient` records previously
persisted by :class:`~dattri_llm.gradient.file_manager.GradientFileManager`; no
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
K-FAC score.  Layers whose parameter count exceeds ``direct_fim_max_params`` are
left out to bound the ``O(d^2)`` Fisher; embedding layers (heavily parametrised)
stay ignored.
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
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.streaming import (
    DiskGradientSource,
    GradientSource,
    GradientStreamer,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig


def _select_layers(grad: Gradient, predicate: Callable[[str], bool]) -> list[str]:
    """Names of layers in *grad* whose type satisfies *predicate*."""
    return [name for name in grad.data if predicate(grad.layer_types[name])]


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
            GradientFileManager(train_dir),
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
            GradientFileManager(test_dir),
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
        fisher_acc: ops.FisherAccumulator | None,
        damping: float,
    ) -> object:
        """Estimate the per-layer K-FAC preconditioner from the training gradients.

        Iterates ``train_source`` (a re-iterable ``GradientSource`` -- disk or a
        frozen streamer; EK-FAC iterates it twice).  Returns an opaque context
        object (possibly empty if no K-FAC-eligible layer is present) passed back
        to :meth:`_prepare_test` and :meth:`_score`.

        When *fisher_acc* is not ``None`` the implementation must also feed every
        block to it (via :meth:`_accumulate_fisher`) during its **first** pass, so
        the direct-Fisher estimate for non-K-FAC layers reuses that sweep.
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

    def _run(
        self,
        train_source: GradientSource,
        test_source: GradientSource,
        *,
        damping: float = 1e-3,
        loop_over_test: bool = False,
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
        """
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}.")
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

        # Fit the K-FAC preconditioner over the training gradients.  When the
        # direct strategy is requested, an empirical-Fisher accumulator for the
        # non-K-FAC (norm) layers is filled in the *same* first pass.
        self._fisher_saw_embedding = set()
        fisher_acc = (
            ops.FisherAccumulator(direct_fim_max_params)
            if non_kfac_strategy == "direct"
            else None
        )
        ctx = self._fit(train_source, device, fisher_acc, damping)
        fim_ctx: dict[str, torch.Tensor] = (
            self._finalize_fisher(fisher_acc, direct_fim_max_params, damping)
            if fisher_acc is not None
            else {}
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

        # Per test block: (K-FAC rep, direct-Fisher rep).  Per (train, test) pair:
        # the summed K-FAC + direct-Fisher score.
        def prepare(test_g: Gradient) -> tuple[object, dict[str, torch.Tensor]]:
            return self._prepare_test(test_g, ctx), self._prepare_fim_test(
                test_g,
                fim_ctx,
            )

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
                :class:`GradientFileManager` for the train pass.
            test_gradients_dir: Directory written by
                :class:`GradientFileManager` for the test pass.
            damping: Tikhonov term added to each covariance factor (K-FAC) or
                to the corrected eigenvalues (EK-FAC) before inversion.
            selected_training_steps: Restrict the train checkpoints (Fisher fit +
                output rows) to these steps; ``None`` uses all on disk.
            loop_over_test: Re-stream + rebuild the test reps per train block (low
                memory) instead of caching them once (default).
            verbose: Show tqdm progress bars on the logging process.
            non_kfac_strategy: ``"ignore"`` (default) skips norm layers; ``"direct"``
                adds a dense empirical-Fisher preconditioner for each, bounded by
                ``direct_fim_max_params``.
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
        train = DiskGradientSource(
            GradientFileManager(train_gradients_dir),
            self.args,
            steps=selected_training_steps,
            layer_name=layer_name,
            desc=f"{self.algorithm}: train",
            verbose=verbose,
        )
        test = DiskGradientSource(
            GradientFileManager(test_gradients_dir),
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
            non_kfac_strategy=non_kfac_strategy,
            direct_fim_max_params=direct_fim_max_params,
            algorithm_meta_extra={"selected_training_steps": train._steps},
            layer_name=layer_name,
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

    @staticmethod
    def _kfac_layers(grad: Gradient) -> list[str]:
        """Layer names eligible for K-FAC (linear/conv)."""
        return _select_layers(
            grad,
            lambda lt: (
                ops.is_linear(lt) or ops.is_conv(lt) or ops.is_conv_transpose(lt)
            ),
        )

    @staticmethod
    def _fisher_layers(grad: Gradient) -> list[str]:
        """Layer names eligible for the direct-Fisher fallback (norm).

        Symmetric to :meth:`_kfac_layers`.
        """
        return _select_layers(grad, ops.is_norm)

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
        self._fisher_saw_embedding.update(
            _select_layers(grad, ops.is_embedding),
        )

    def _finalize_fisher(
        self,
        fisher_acc: ops.FisherAccumulator,
        max_params: int,
        damping: float,
    ) -> dict[str, torch.Tensor]:
        """Turn the accumulated Fishers into ``{layer: F_l^-1}``, warning about the
        norm layers dropped by the ``max_params`` cap and the ignored embeddings.
        """
        if fisher_acc.skipped:
            warnings.warn(
                "non_kfac_strategy='direct' skipped norm layers whose parameter "
                f"count exceeds direct_fim_max_params={max_params}: "
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
        """Per-sample ``(B, P)`` weight gradient used for direct-Fisher scoring."""
        return ops.materialize(grad.data[layer], grad.layer_types[layer])

    def _prepare_fim_test(
        self,
        test_g: Gradient,
        fim_ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``(B_te, P)`` weight gradient per direct-Fisher layer."""
        return {
            layer: self._fisher_grad(test_g, layer)
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

        ``F_l^-1`` is symmetric, so ``score = M_tr F^-1 M_te^T``.
        """
        total: torch.Tensor | None = None
        for layer, F_inv in fim_ctx.items():
            if layer not in train_g.data or layer not in fim_test_rep:
                continue
            M_tr = self._fisher_grad(train_g, layer).float()  # (B_tr, P)
            M_te = fim_test_rep[layer].float()  # (B_te, P)
            block = (M_tr @ F_inv) @ M_te.T  # (B_tr, B_te)
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
        fisher_acc: ops.FisherAccumulator | None,
        damping: float,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            # Reuse this single sweep to fit the direct Fisher for norm layers.
            if fisher_acc is not None:
                self._accumulate_fisher(fisher_acc, train_g)
        return {
            layer: (
                ops.sym_inverse(A, damping),
                ops.sym_inverse(G, damping),
            )
            for layer, (A, G) in kron.result().items()
        }

    @staticmethod
    def _prepare_test(test_g: Gradient, _ctx: object) -> Gradient:
        # K-FAC's factorised cross-gram reads the raw factors directly, so the
        # test "representation" is just the (device-resident) gradient block.
        return test_g

    @staticmethod
    def _score(
        train_g: Gradient,
        test_g: Gradient,
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for layer, (A_inv, G_inv) in ctx.items():
            if layer not in train_g.data or layer not in test_g.data:
                continue
            block = ops.kfac_cross(
                train_g.data[layer],
                test_g.data[layer],
                train_g.layer_types[layer],
                A_inv,
                G_inv,
            )
            total = block if total is None else total + block
        if total is None:
            total = torch.zeros(train_g.batch_size, test_g.batch_size)
        return total


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
        fisher_acc: ops.FisherAccumulator | None,
        damping: float,
    ) -> dict:
        # Pass 1 -- Kronecker covariance factors and their eigenbases (and, when
        # requested, the direct Fisher for norm layers from the same sweep).
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            if fisher_acc is not None:
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
        # Rotate + materialise the test gradients into the eigenbasis once; the
        # result (one ``(B_te, D)`` matrix per layer) is what each train block is
        # contracted against.
        return {
            layer: ops.ekfac_materialize(
                test_g.data[layer],
                test_g.layer_types[layer],
                U_A,
                U_G,
            )
            for layer, (U_A, U_G, _) in ctx.items()
            if layer in test_g.data
        }

    @staticmethod
    def _score(
        train_g: Gradient,
        test_mats: dict[str, torch.Tensor],
        ctx: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for layer, (U_A, U_G, lam) in ctx.items():
            if layer not in train_g.data or layer not in test_mats:
                continue
            M_tr = ops.ekfac_materialize(
                train_g.data[layer],
                train_g.layer_types[layer],
                U_A,
                U_G,
            )  # (B_tr, D)
            M_te = test_mats[layer]  # (B_te, D)
            block = (M_tr / lam) @ M_te.T  # lam is damped at fit time; (B_tr, B_te)
            total = block if total is None else total + block
        if total is None:
            total = torch.zeros(
                train_g.batch_size,
                next(iter(test_mats.values())).shape[0] if test_mats else 0,
            )
        return total
