"""K-FAC / EK-FAC influence attribution over on-disk gradients (workflow 2).

Like :class:`~dattri_llm.algorithm.tracin.TracInAttributor`, these attributors
consume :class:`~dattri_llm.gradient.gradient.Gradient` records previously
persisted by :class:`~dattri_llm.gradient.file_manager.GradientFileManager`; no
forward/backward pass is run at attribution time.  Unlike TracIn (a raw inner
product), they precondition the inner product by an approximate inverse Fisher
estimated *from the training gradients themselves*.

These are **single-checkpoint** methods: there is no per-step ensemble (no
``steps``/``weights``).  Every record in ``train_gradients_dir`` is one training
sample and every record in ``test_gradients_dir`` is one test sample; the score
is the full ``(num_train, num_test)`` matrix

    score[i, j] = Σ_layer  vec(∇W_te,j)ᵀ F_l⁻¹ vec(∇W_tr,i)

where the per-layer Fisher is approximated with the Kronecker structure
``F_l ≈ A_l ⊗ G_l`` (``A`` the input-activation covariance, ``G`` the
output-gradient covariance) fit over the whole training set.

* **K-FAC** uses ``F_l⁻¹ ≈ (A_l + λ)⁻¹ ⊗ (G_l + λ)⁻¹``.  Because the per-sample
  gradient factorises as ``Σ_t g_t a_tᵀ``, this collapses to a whitened version
  of the factorised cross-gram — no weight gradient is ever materialised.
* **EK-FAC** rotates into the Kronecker eigenbasis ``U_A, U_G`` and replaces the
  Kronecker eigenvalues with the *empirical* second moments ``Λ`` of the
  projected gradients (a second pass over the training gradients), giving
  ``F_l⁻¹ ≈ (U_A ⊗ U_G) (Λ + λ)⁻¹ (U_A ⊗ U_G)ᵀ``.

Only linear and convolution layers are K-FAC-eligible; normalisation and
embedding layers (for which K-FAC is undefined) are skipped by default.  Token/
spatial positions are summed (matching a sum-over-tokens loss).  Rows and columns
of the returned :class:`~dattri_llm.algorithm.score.AttributionScore` are
identified by the on-disk content hash, in disk order.

Normalisation layers are not heavily parametrised, so their per-layer Fisher can
be estimated **directly** rather than with the Kronecker factorisation.  Passing
``non_kfac_strategy="direct"`` to :meth:`attribute_from_cache` adds a dense
empirical-Fisher preconditioner ``F_l⁻¹`` for each such layer (built from the
token-summed ``(B, d)`` weight gradients), whose contribution is summed into the
K-FAC score.  Layers whose parameter count exceeds ``direct_fim_max_params`` are
left out to bound the ``O(d²)`` Fisher; embedding layers (heavily parametrised)
stay ignored.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import (
    BaseAttributor,
    iter_gradient_blocks,
    resolve_steps,
)
from dattri_llm.algorithm.score import AttributionScore
from dattri_llm.algorithm.task import AttributionTask
from dattri_llm.gradient import ops
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient
from torch.utils.data import Dataset


def _select_layers(
    grad: Gradient, layer_name: Optional[List[str]], predicate
) -> List[str]:
    """Names of layers in *grad* whose type satisfies *predicate*, restricted to
    *layer_name* when given."""
    return [
        name
        for name in grad.data
        if predicate(grad.layer_types[name])
        and (layer_name is None or name in layer_name)
    ]


class _KroneckerBaseAttributor(BaseAttributor):
    """Shared on-disk plumbing for the K-FAC family.

    Subclasses implement :meth:`_fit` (estimate the per-layer preconditioner from
    the whole training set), :meth:`_prepare_test` (build a test block's scoring
    representation), and :meth:`_score` (preconditioned cross-gram for a train
    block vs a prepared test rep).  Everything else — file-granular block
    iteration, ``loop_over_test`` memory control, column bookkeeping, and the
    ``(num_train, num_test)`` assembly — is shared.
    """

    algorithm: str = "Kronecker"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        damping: float = 1e-3,
        layer_name: Optional[Union[str, List[str]]] = None,
        task: Optional[AttributionTask] = None,
    ) -> None:
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}.")
        self.args = args
        self.task = task
        self.damping = float(damping)
        if isinstance(layer_name, str):
            self.layer_name: Optional[List[str]] = [layer_name]
        elif layer_name is None:
            self.layer_name = None
        else:
            self.layer_name = list(layer_name)

    def cache(self, train_dataset: Dataset) -> None:
        """No-op: gradients are already cached on disk in this workflow."""

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _fit(
        self,
        train_fm: GradientFileManager,
        train_steps: List[int],
        device: torch.device,
        verbose: bool,
        fisher_acc: Optional[ops.FisherAccumulator],
    ) -> object:
        """Estimate the per-layer K-FAC preconditioner from the selected training
        gradients (the records at *train_steps*).

        Returns an opaque context object (possibly empty if no K-FAC-eligible
        layer is present) passed back to :meth:`_prepare_test` and :meth:`_score`.

        When *fisher_acc* is not ``None`` the implementation must also feed every
        streamed block to it (via :meth:`_accumulate_fisher`) during its **first**
        pass, so the direct-Fisher estimate for non-K-FAC layers reuses the same
        single sweep over the data.
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
        self, train_g: Gradient, test_rep: object, ctx: object
    ) -> torch.Tensor:
        """``(B_train, B_test)`` preconditioned score for a train block against a
        test representation from :meth:`_prepare_test`."""

    # ------------------------------------------------------------------ #
    # Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def attribute_from_cache(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        loop_over_test: bool = False,
        selected_training_steps: Optional[Iterable[int]] = None,
        verbose: bool = False,
        non_kfac_strategy: Literal["ignore", "direct"] = "ignore",
        direct_fim_max_params: int = 4096,
    ) -> AttributionScore:
        """Compute the ``(num_train, num_test)`` attribution score from on-disk
        gradients — every train record against every test record.

        Args:
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during the training pass.  Also the
                data the Fisher is estimated from.
            test_gradients_dir: Directory written during the test pass.
            loop_over_test: If ``False`` (default), the test representations are
                prepared once and reused across all train blocks (peak memory:
                all test reps + one train block).  If ``True``, they are
                re-streamed and rebuilt for every train block (peak memory: one
                train + one test block) at the cost of more disk reads — use this
                when the test set does not fit in memory.
            selected_training_steps: Restrict the training checkpoints used (the
                Fisher fit and the output rows) to these steps; ``None`` (default)
                uses every step on disk.  Over-specified ranges are intersected
                with what is available.  The test set always supplies every
                column.
            verbose: Show tqdm progress bars on the logging process.
            non_kfac_strategy: How to treat non-K-FAC layers (normalisation
                layers).  ``"ignore"`` (default) skips them — only linear/conv
                layers contribute.  ``"direct"`` adds a dense empirical-Fisher
                preconditioner for each such layer and sums its contribution into
                the score.
            direct_fim_max_params: Only used when ``non_kfac_strategy="direct"``.
                A non-K-FAC layer is included only if its (token-summed) weight
                gradient has at most this many parameters, bounding the dense
                ``O(d²)`` Fisher.  Larger layers are skipped with a warning.

        Returns:
            An :class:`AttributionScore` whose rows/columns are the train/test
            content hashes (disk order); also persisted to ``args.output_dir``.

        Raises:
            ValueError: If a gradients dir is missing, ``selected_training_steps``
                matches no step in the training gradients, no eligible layers are
                present in the training gradients, or an argument is invalid.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__} requires both train_gradients_dir and "
                "test_gradients_dir."
            )
        if non_kfac_strategy not in ("ignore", "direct"):
            raise ValueError(
                "non_kfac_strategy must be 'ignore' or 'direct', got "
                f"{non_kfac_strategy!r}."
            )
        if direct_fim_max_params <= 0:
            raise ValueError(
                f"direct_fim_max_params must be positive, got {direct_fim_max_params}."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)
        device = self.args.device

        # ``selected_training_steps`` picks which training checkpoints to
        # attribute from (the Fisher fit and the output rows).  The test set
        # defines the query columns and is always used in full.
        train_steps = resolve_steps(train_fm, selected_training_steps)
        test_steps = test_fm.available_steps()

        # Fit the K-FAC preconditioner over the selected training gradients.  When
        # the direct strategy is requested, an empirical-Fisher accumulator for the
        # non-K-FAC (norm) layers is filled in the *same* first pass, so the data
        # is streamed only once.
        self._fisher_saw_embedding = set()
        fisher_acc = (
            ops.FisherAccumulator(direct_fim_max_params)
            if non_kfac_strategy == "direct"
            else None
        )
        ctx = self._fit(train_fm, train_steps, device, verbose, fisher_acc)
        fim_ctx: Dict[str, torch.Tensor] = (
            self._finalize_fisher(fisher_acc, direct_fim_max_params)
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
                + ". Check `layer_name` and the collected layers."
            )

        def _test_reps(td: Gradient) -> Tuple[object, Dict[str, torch.Tensor]]:
            return self._prepare_test(td, ctx), self._prepare_fim_test(td, fim_ctx)

        # One pass over the test files: fix the column order (disk order) and,
        # unless looping, build + cache each block's test representation.
        test_ids: List[str] = []
        test_index: dict = {}
        cached_test: List[Tuple[object, Dict[str, torch.Tensor], List[str]]] = []
        for _step, test_g, test_hashes in iter_gradient_blocks(
            test_fm, test_steps, self.args, self.layer_name,
            desc=f"{self.algorithm}: preparing test",
            verbose=verbose,
        ):
            for h in test_hashes:
                if h not in test_index:
                    test_index[h] = len(test_ids)
                    test_ids.append(h)
            if not loop_over_test:
                kfac_rep, fim_rep = _test_reps(test_g.to(device))
                cached_test.append((kfac_rep, fim_rep, test_hashes))
        num_test = len(test_ids)

        def _loop_test_reps():
            for _s, g, h in iter_gradient_blocks(
                test_fm, test_steps, self.args, self.layer_name
            ):
                kfac_rep, fim_rep = _test_reps(g.to(device))
                yield kfac_rep, fim_rep, h

        # Score every selected train block against every test representation.
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        for train_step, train_g, train_hashes in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc=f"{self.algorithm}: scoring",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            row = torch.zeros(train_g.batch_size, num_test, dtype=torch.float)
            test_reps = _loop_test_reps() if loop_over_test else cached_test
            for test_rep, fim_rep, test_hashes in test_reps:
                cols = [test_index[h] for h in test_hashes]
                block = self._combine_score(
                    train_g, test_rep, fim_rep, ctx, fim_ctx, len(cols)
                )
                row[:, cols] = block.detach().to("cpu", torch.float)
            row_chunks.append(row)
            row_train_ids.extend(train_hashes)
            row_steps.extend([train_step] * train_g.batch_size)

        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, num_test, dtype=torch.float)
        )

        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            # One row per train sample, stamped with the step it was recorded at.
            # These are single-checkpoint methods, so a given dir typically holds
            # one step; the real step is preserved rather than forced to 0.
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta={
                "damping": self.damping,
                "selected_training_steps": train_steps,
                "non_kfac_strategy": non_kfac_strategy,
                "direct_fim_layers": sorted(fim_ctx),
            },
            algorithm=self.algorithm,
            normalized_grad=False,
            layer_name=self.layer_name,
        )
        result.save(self.args.output_path)
        return result

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _kfac_layers(self, grad: Gradient) -> List[str]:
        """Layer names eligible for K-FAC (linear/conv), honouring ``layer_name``."""
        return _select_layers(
            grad,
            self.layer_name,
            lambda lt: ops.is_linear(lt) or ops.is_conv(lt) or ops.is_conv_transpose(lt),
        )

    def _fisher_layers(self, grad: Gradient) -> List[str]:
        """Layer names eligible for the direct-Fisher fallback (norm), honouring
        ``layer_name``.  Symmetric to :meth:`_kfac_layers`."""
        return _select_layers(grad, self.layer_name, ops.is_norm)

    # ------------------------------------------------------------------ #
    # Direct-Fisher fallback for non-K-FAC (norm) layers                  #
    # ------------------------------------------------------------------ #

    def _accumulate_fisher(
        self, fisher_acc: ops.FisherAccumulator, grad: Gradient
    ) -> None:
        """Fold one streamed training block into the per-layer Fisher estimate.

        Called from each subclass's :meth:`_fit` first pass.  Also records any
        embedding layers seen so :meth:`_finalize_fisher` can warn that they were
        left ignored (heavily parametrised — not covered by the direct fallback).
        """
        fisher_acc.update(grad, self._fisher_layers(grad))
        self._fisher_saw_embedding.update(
            _select_layers(grad, self.layer_name, ops.is_embedding)
        )

    def _finalize_fisher(
        self, fisher_acc: ops.FisherAccumulator, max_params: int
    ) -> Dict[str, torch.Tensor]:
        """Turn the accumulated Fishers into ``{layer: F_l⁻¹}``, warning about the
        norm layers dropped by the ``max_params`` cap and the ignored embeddings."""
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
            layer: ops.sym_inverse(F, self.damping)
            for layer, F in fisher_acc.result().items()
        }

    @staticmethod
    def _fisher_grad(grad: Gradient, layer: str) -> torch.Tensor:
        """Per-sample ``(B, P)`` weight gradient used for direct-Fisher scoring."""
        f = grad.data[layer]
        return ops.weight_grad(
            f.activation, f.pre_activation_grad, grad.layer_types[layer], f.module_kwargs
        )

    def _prepare_fim_test(
        self, test_g: Gradient, fim_ctx: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """``(B_te, P)`` weight gradient per direct-Fisher layer."""
        return {
            layer: self._fisher_grad(test_g, layer)
            for layer in fim_ctx
            if layer in test_g.data
        }

    def _score_fim(
        self,
        train_g: Gradient,
        fim_test_rep: Dict[str, torch.Tensor],
        fim_ctx: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """``(B_tr, B_te)`` direct-Fisher score, or ``None`` if no layer applies.

        ``F_l⁻¹`` is symmetric, so ``score = M_tr F⁻¹ M_teᵀ``.
        """
        total: Optional[torch.Tensor] = None
        for layer, F_inv in fim_ctx.items():
            if layer not in train_g.data or layer not in fim_test_rep:
                continue
            M_tr = self._fisher_grad(train_g, layer).float()     # (B_tr, P)
            M_te = fim_test_rep[layer].float()                   # (B_te, P)
            block = (M_tr @ F_inv) @ M_te.T                      # (B_tr, B_te)
            total = block if total is None else total + block
        return total

    def _combine_score(
        self,
        train_g: Gradient,
        test_rep: object,
        fim_test_rep: Dict[str, torch.Tensor],
        ctx: object,
        fim_ctx: Dict[str, torch.Tensor],
        n_test: int,
    ) -> torch.Tensor:
        """Sum the K-FAC and direct-Fisher contributions for one train/test pair.

        Either side may be empty (no eligible layers); the other carries the
        score.  At least one is non-empty (enforced by ``attribute_from_cache``);
        *n_test* is the test block's column count, used only when neither side
        shares a layer with this block (an all-zero contribution).
        """
        block: Optional[torch.Tensor] = None
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

    ``F_l⁻¹ ≈ (A_l + λ)⁻¹ ⊗ (G_l + λ)⁻¹`` per linear/conv layer, with ``λ`` the
    ``damping`` term.  See the module docstring for the score definition.

    Args:
        args: :class:`AttributionArguments` (``dataloader_*``, ``device``,
            ``output_dir`` are consulted).
        damping: Tikhonov term added to each covariance factor before inversion.
        layer_name: Restrict to these layers; ``None`` uses every eligible layer.
        task: Accepted for API parity; unused.

    The training checkpoints used are chosen per call via
    :meth:`attribute`'s ``selected_training_steps`` argument.
    """

    algorithm = "KFAC"

    def _fit(
        self,
        train_fm: GradientFileManager,
        train_steps: List[int],
        device: torch.device,
        verbose: bool,
        fisher_acc: Optional[ops.FisherAccumulator],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        kron = ops.KroneckerAccumulator()
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="KFAC: fitting",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            # Reuse this single sweep to fit the direct Fisher for norm layers.
            if fisher_acc is not None:
                self._accumulate_fisher(fisher_acc, train_g)
        return {
            layer: (
                ops.sym_inverse(A, self.damping),
                ops.sym_inverse(G, self.damping),
            )
            for layer, (A, G) in kron.result().items()
        }

    def _prepare_test(self, test_g: Gradient, ctx: object) -> Gradient:
        # K-FAC's factorised cross-gram reads the raw factors directly, so the
        # test "representation" is just the (device-resident) gradient block.
        return test_g

    def _score(
        self, train_g: Gradient, test_g: Gradient,
        ctx: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for layer, (A_inv, G_inv) in ctx.items():
            if layer not in train_g.data or layer not in test_g.data:
                continue
            tf = train_g.data[layer]
            ef = test_g.data[layer]
            block = ops.kfac_cross(
                tf.activation, tf.pre_activation_grad,
                ef.activation, ef.pre_activation_grad,
                train_g.layer_types[layer], A_inv, G_inv,
                tf.module_kwargs, ef.module_kwargs,
            )
            total = block if total is None else total + block
        if total is None:
            total = torch.zeros(train_g.batch_size, test_g.batch_size)
        return total


class EKFACAttributor(_KroneckerBaseAttributor):
    """EK-FAC influence attributor.

    Rotates each layer's gradients into the Kronecker eigenbasis ``(U_A, U_G)``
    and replaces the Kronecker eigenvalues with the empirical second moments
    ``Λ`` of the projected gradients (a second pass over the training
    gradients), giving ``F_l⁻¹ ≈ (U_A ⊗ U_G)(Λ + λ)⁻¹(U_A ⊗ U_G)ᵀ``.

    The per-sample gradient is projected as ``M = U_Gᵀ ∇W U_A`` — the faithful
    expansion of ``(U_A ⊗ U_G)ᵀ vec(∇W)``.  This is the unique projection that
    **reduces to K-FAC** when ``Λ`` equals the Kronecker eigenvalues, and it is
    invariant to the (arbitrary) sign of each eigenvector.

    ``mode`` selects the implementation and is kept mainly for backward
    compatibility / cross-checking:

    * ``"exact"`` *(default)* — the faithful projection above.
    * ``"approx"`` — the code path mirroring the ``dattri`` library.  Its
      original transposed projection ``U_G ∇W U_Aᵀ`` was wrong (does not reduce
      to K-FAC and is sign-sensitive; see
      ``test_transposed_projection_is_sign_sensitive``); fixed, it now uses the
      same faithful projection, so the two modes produce identical scores.

    Args:
        args: :class:`AttributionArguments`.
        damping: Added to the corrected eigenvalues before inversion.
        layer_name: Restrict to these layers; ``None`` uses every eligible layer.
        task: Accepted for API parity; unused.
        mode: ``"exact"`` (default) or ``"approx"``; see above.  Currently
            equivalent.

    The training checkpoints used are chosen per call via
    :meth:`attribute`'s ``selected_training_steps`` argument.
    """

    algorithm = "EKFAC"
    EKFAC_MODES = ("exact", "approx")

    def __init__(
        self,
        args: AttributionArguments,
        *,
        damping: float = 1e-3,
        layer_name: Optional[Union[str, List[str]]] = None,
        task: Optional[AttributionTask] = None,
        mode: str = "exact",
    ) -> None:
        if mode not in self.EKFAC_MODES:
            raise ValueError(
                f"mode must be one of {self.EKFAC_MODES}, got {mode!r}."
            )
        super().__init__(
            args, damping=damping, layer_name=layer_name, task=task
        )
        self.mode = mode

    def _fit(
        self,
        train_fm: GradientFileManager,
        train_steps: List[int],
        device: torch.device,
        verbose: bool,
        fisher_acc: Optional[ops.FisherAccumulator],
    ) -> dict:
        # Pass 1 — Kronecker covariance factors and their eigenbases (and, when
        # requested, the direct Fisher for norm layers from the same sweep).
        kron = ops.KroneckerAccumulator()
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="EKFAC: fitting factors",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            kron.update(train_g, self._kfac_layers(train_g))
            if fisher_acc is not None:
                self._accumulate_fisher(fisher_acc, train_g)
        # Eigenvectors are fed to ``ekfac_materialize`` (which does ``a @ U``),
        # giving the faithful projection ``M = U_Gᵀ ∇W U_A``.
        eig: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, (A, G) in kron.result().items():
            _, U_A, _, U_G = ops.kfac_eigh(A, G)
            if self.mode == "approx":
                # 'approx' mirrors the dattri code path.  Its transpose here
                # (U_A.T, U_G.T) was the bug; fixed, it uses the faithful
                # eigenvectors, matching 'exact'.
                U_A, U_G = U_A, U_G
            eig[layer] = (U_A, U_G)

        # Pass 2 — empirical second moments of the projected gradients (Λ).
        # Skipped entirely when no K-FAC layer is present (norm-only + direct),
        # so the data is not streamed for nothing.
        lam_sum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="EKFAC: fitting correction",
            verbose=verbose,
        ) if eig else ():
            train_g = train_g.to(device)
            for layer, (U_A, U_G) in eig.items():
                if layer not in train_g.data:
                    continue
                f = train_g.data[layer]
                M = ops.ekfac_materialize(
                    f.activation, f.pre_activation_grad,
                    train_g.layer_types[layer], U_A, U_G, f.module_kwargs,
                )  # (B, D)
                lam_sum[layer] = lam_sum.get(layer, 0) + (M * M).sum(0)
                counts[layer] = counts.get(layer, 0) + M.shape[0]

        layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for layer, (U_A, U_G) in eig.items():
            layers[layer] = (U_A, U_G, lam_sum[layer] / counts[layer])
        return layers

    def _prepare_test(
        self, test_g: Gradient, ctx: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Rotate + materialise the test gradients into the eigenbasis once; the
        # result (one ``(B_te, D)`` matrix per layer) is what each train block is
        # contracted against.
        return {
            layer: ops.ekfac_materialize(
                test_g.data[layer].activation,
                test_g.data[layer].pre_activation_grad,
                test_g.layer_types[layer], U_A, U_G,
                test_g.data[layer].module_kwargs,
            )
            for layer, (U_A, U_G, _) in ctx.items()
            if layer in test_g.data
        }

    def _score(
        self, train_g: Gradient, test_mats: Dict[str, torch.Tensor],
        ctx: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for layer, (U_A, U_G, lam) in ctx.items():
            if layer not in train_g.data or layer not in test_mats:
                continue
            tf = train_g.data[layer]
            M_tr = ops.ekfac_materialize(
                tf.activation, tf.pre_activation_grad,
                train_g.layer_types[layer], U_A, U_G, tf.module_kwargs,
            )  # (B_tr, D)
            M_te = test_mats[layer]  # (B_te, D)
            block = (M_tr / (lam + self.damping)) @ M_te.T  # (B_tr, B_te)
            total = block if total is None else total + block
        if total is None:
            total = torch.zeros(train_g.batch_size, next(iter(test_mats.values())).shape[0]
                                if test_mats else 0)
        return total
