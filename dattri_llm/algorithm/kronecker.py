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
embedding layers (for which K-FAC is undefined) are skipped.  Token/spatial
positions are summed (matching a sum-over-tokens loss).  Rows and columns of the
returned :class:`~dattri_llm.algorithm.score.AttributionScore` are identified by
the on-disk content hash, in disk order.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
    ) -> object:
        """Estimate the per-layer preconditioner from the selected training
        gradients (the records at *train_steps*).

        Returns an opaque context object passed back to :meth:`_prepare_test`
        and :meth:`_score`.
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

        Returns:
            An :class:`AttributionScore` whose rows/columns are the train/test
            content hashes (disk order); also persisted to ``args.output_dir``.

        Raises:
            ValueError: If a gradients dir is missing, ``selected_training_steps``
                matches no step in the training gradients, or no K-FAC-eligible
                layers are present in the training gradients.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__} requires both train_gradients_dir and "
                "test_gradients_dir."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)
        device = self.args.device

        # ``selected_training_steps`` picks which training checkpoints to
        # attribute from (the Fisher fit and the output rows).  The test set
        # defines the query columns and is always used in full.
        train_steps = resolve_steps(train_fm, selected_training_steps)
        test_steps = test_fm.available_steps()

        # Fit the preconditioner once over the selected training gradients.
        ctx = self._fit(train_fm, train_steps, device, verbose)

        # One pass over the test files: fix the column order (disk order) and,
        # unless looping, build + cache each block's test representation.
        test_ids: List[str] = []
        test_index: dict = {}
        cached_test: List[Tuple[object, List[str]]] = []
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
                cached_test.append((self._prepare_test(test_g.to(device), ctx), test_hashes))
        num_test = len(test_ids)

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
            test_reps = (
                (
                    (self._prepare_test(g.to(device), ctx), h)
                    for _s, g, h in iter_gradient_blocks(
                        test_fm, test_steps, self.args, self.layer_name
                    )
                )
                if loop_over_test
                else cached_test
            )
            for test_rep, test_hashes in test_reps:
                cols = [test_index[h] for h in test_hashes]
                block = self._score(train_g, test_rep, ctx)
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
        out = []
        for name in grad.data:
            lt = grad.layer_types[name]
            if not (ops.is_linear(lt) or ops.is_conv(lt) or ops.is_conv_transpose(lt)):
                continue
            if self.layer_name is not None and name not in self.layer_name:
                continue
            out.append(name)
        return out


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
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        accums: Dict[str, ops.KFACAccumulator] = {}
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="KFAC: fitting",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            for layer in self._kfac_layers(train_g):
                f = train_g.data[layer]
                accums.setdefault(layer, ops.KFACAccumulator()).update(
                    f.activation, f.pre_activation_grad,
                    train_g.layer_types[layer], f.module_kwargs,
                )
        if not accums:
            raise ValueError(
                "No K-FAC-eligible (linear/conv) layers found in the training "
                "gradients; check `layer_name` and the collected layers."
            )
        precond: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, acc in accums.items():
            A, G = acc.result()
            precond[layer] = (
                ops.sym_inverse(A, self.damping),
                ops.sym_inverse(G, self.damping),
            )
        return precond

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
    ) -> dict:
        # Pass 1 — Kronecker covariance factors and their eigenbases.
        accums: Dict[str, ops.KFACAccumulator] = {}
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="EKFAC: fitting factors",
            verbose=verbose,
        ):
            train_g = train_g.to(device)
            for layer in self._kfac_layers(train_g):
                f = train_g.data[layer]
                accums.setdefault(layer, ops.KFACAccumulator()).update(
                    f.activation, f.pre_activation_grad,
                    train_g.layer_types[layer], f.module_kwargs,
                )
        if not accums:
            raise ValueError(
                "No K-FAC-eligible (linear/conv) layers found in the training "
                "gradients; check `layer_name` and the collected layers."
            )
        # Eigenvectors are fed to ``ekfac_materialize`` (which does ``a @ U``),
        # giving the faithful projection ``M = U_Gᵀ ∇W U_A``.
        eig: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, acc in accums.items():
            A, G = acc.result()
            _, U_A, _, U_G = ops.kfac_eigh(A, G)
            if self.mode == "approx":
                # 'approx' mirrors the dattri code path.  Its transpose here
                # (U_A.T, U_G.T) was the bug; fixed, it uses the faithful
                # eigenvectors, matching 'exact'.
                U_A, U_G = U_A, U_G
            eig[layer] = (U_A, U_G)

        # Pass 2 — empirical second moments of the projected gradients (Λ).
        lam_sum: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for _step, train_g, _ in iter_gradient_blocks(
            train_fm, train_steps, self.args, self.layer_name,
            desc="EKFAC: fitting correction",
            verbose=verbose,
        ):
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
