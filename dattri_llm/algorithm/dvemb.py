"""DVEmb (Data Value Embedding) trajectory-aware attribution (workflow 2).

Like :class:`~dattri_llm.algorithm.tracin.TracInAttributor` and the K-FAC
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

    H_k ≈ Σ_{z ∈ B_k} ∇ℓ(θ_k, z) ∇ℓ(θ_k, z)ᵀ.                              (2)

Setting every ``H_k = 0`` recovers TracIn (η · ⟨g_test, g_train⟩); the Fisher
factors are exactly the "training dynamics" correction DVEmb adds.

**Computation.** Because each ``H_k`` is symmetric, the product in (1) is
applied to the *test* side by sweeping the recorded training steps from latest
to earliest.  For every test column we carry a running parameter-space vector

    w_{t_s} = [ ∏_{k=t_s+1}^{T-1}(I − η_k H_k) ]ᵀ ∇ℓ(θ_T, z_val),

initialised at the final-model test gradient.  At each step ``t_s`` (descending)
the rows for the train samples recorded there are ``η_{t_s} · ⟨g(z*), w_{t_s}⟩``,
after which ``w`` is advanced by that step's full Fisher factor
``w ← w − η_{t_s} Σ_{z∈B_{t_s}} g(z) ⟨g(z), w⟩``.  The recorded per-sample
gradients of a step thus serve twice: as the vectors scored, and as the rank-1
components of ``H_{t_s}``.

This is the **basic** DVEmb estimator — it materialises the per-layer gradients
and propagates the exact (Fisher-approximated) product.  Influence-checkpointing
and the low-rank embedding compression of the paper are deliberately omitted.

The result is an :class:`~dattri_llm.algorithm.score.AttributionScore` whose
rows are ``(train_hash, step)`` pairs (one row per recorded checkpoint of a
sample, stamped with its step) and whose columns are the test-sample hashes in
on-disk order — identical bookkeeping to TracIn and the K-FAC family.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from dattri_llm.algorithm.arguments import AttributionArguments
from dattri_llm.algorithm.base import (
    BaseAttributor,
    iter_gradient_blocks,
    resolve_steps,
)
from dattri_llm.algorithm.score import AttributionScore
from dattri_llm.algorithm.task import AttributionTask
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Gradient


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
        learning_rate: The SGD learning rate ``η`` used during training.  Either
            a single float (constant schedule) or a mapping ``{step: η_step}``
            giving the rate at each recorded step.  It enters both the per-step
            score scale ``η_{t_s}`` and the Fisher factors ``(I − η_k H_k)``, so
            it must match the schedule the gradients were collected under.  A
            mapping must cover every propagated step (``step < final_step``).
        layer_name: Restrict the gradients (and therefore the Fisher) to these
            layer names.  ``None`` uses every layer present in the gradients.
        task: Accepted for parity with the workflow-1 API but unused here.
    """

    algorithm = "DVEmb"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        learning_rate: Union[float, Mapping[int, float]] = 1.0,
        layer_name: Optional[Union[str, List[str]]] = None,
        task: Optional[AttributionTask] = None,
    ) -> None:
        if isinstance(learning_rate, Mapping):
            self.learning_rate: Union[float, Dict[int, float]] = {
                int(k): float(v) for k, v in learning_rate.items()
            }
        else:
            lr = float(learning_rate)
            if lr < 0:
                raise ValueError(f"learning_rate must be non-negative, got {lr}.")
            self.learning_rate = lr
        self.args = args
        self.task = task
        if isinstance(layer_name, str):
            self.layer_name: Optional[List[str]] = [layer_name]
        elif layer_name is None:
            self.layer_name = None
        else:
            self.layer_name = list(layer_name)

    def cache(self, train_dataset: Dataset) -> None:
        """No-op: gradients are already cached on disk in this workflow."""

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _lr(self, step: int) -> float:
        """Learning rate ``η`` at *step*."""
        if isinstance(self.learning_rate, dict):
            try:
                return self.learning_rate[step]
            except KeyError:
                raise ValueError(
                    f"learning_rate mapping has no entry for step {step}; it must "
                    f"cover every propagated step (step < final_step). "
                    f"Provided steps: {sorted(self.learning_rate)}."
                ) from None
        return self.learning_rate

    @staticmethod
    def _materialize(block: Gradient, device: torch.device) -> Dict[str, torch.Tensor]:
        """Materialise a gradient block into ``{layer: (B, d) float tensor}``."""
        mat = block.to(device).materialize()
        return {name: value.float() for name, value in mat.data.items()}

    # ------------------------------------------------------------------ #
    # Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        train_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        train_gradients_dir: Optional[str] = None,
        test_gradients_dir: Optional[str] = None,
        loop_over_test: bool = False,
        selected_training_steps: Optional[Iterable[int]] = None,
        final_step: Optional[int] = None,
    ) -> AttributionScore:
        """Compute the ``(num_train_rows, num_test)`` DVEmb attribution score.

        Args:
            train_dataset: Unused; kept for API parity.
            test_dataset: Unused; kept for API parity.
            train_gradients_dir: Directory written by
                :class:`GradientFileManager` during training.  Supplies both the
                scored train gradients and, at every step, the per-sample
                gradients forming that step's Fisher factor.
            test_gradients_dir: Directory written during the test pass.  These
                gradients must have been collected at the **final** model
                ``θ_T`` (capital ``T`` = ``final_step``), since the score dots
                against ``∇ℓ(θ_T, z_val)``.
            loop_over_test: Accepted for API parity with TracIn / K-FAC.  DVEmb
                carries the full per-column test embedding through the whole step
                sweep, so the test state is always resident; both values produce
                identical results.
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

        Returns:
            An :class:`AttributionScore`; also persisted to ``args.output_dir``.

        Raises:
            ValueError: If a gradients dir is missing,
                ``selected_training_steps`` matches no available step, or no
                training step satisfies ``step < final_step``.
        """
        if train_gradients_dir is None or test_gradients_dir is None:
            raise ValueError(
                f"{type(self).__name__} requires both train_gradients_dir and "
                "test_gradients_dir."
            )

        train_fm = GradientFileManager(train_gradients_dir)
        test_fm = GradientFileManager(test_gradients_dir)
        device = self.args.device

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
        # ``selected_training_steps`` filters only which steps are emitted as
        # rows; the propagation below always sweeps every step in ``prop_steps``.
        if selected_training_steps is None:
            output_steps = set(prop_steps)
        else:
            output_steps = set(resolve_steps(train_fm, selected_training_steps)) & set(
                prop_steps
            )

        test_steps = test_fm.available_steps()

        # ---- Initialise w = test gradient at the final model θ_T -------- #
        # One pass over the test files fixes the column order (disk order) and
        # materialises each test gradient into the running embedding ``w``,
        # keyed per layer with shape (num_test, d_layer).
        test_ids: List[str] = []
        test_index: Dict[str, int] = {}
        pending: List[Tuple[Dict[str, torch.Tensor], List[int]]] = []
        for _step, test_g, test_hashes in iter_gradient_blocks(
            test_fm, test_steps, self.args, self.layer_name
        ):
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
                # Duplicate-hash columns collapse to the same index; index_copy_
                # keeps the last writer, mirroring on-disk dedup behaviour.
                w[name].index_copy_(0, idx, mat[name])
        del pending

        # ---- Sweep steps latest → earliest, scoring then folding in H --- #
        row_chunks: List[torch.Tensor] = []
        row_train_ids: List[str] = []
        row_steps: List[int] = []
        for ts in sorted(prop_steps, reverse=True):
            lr = self._lr(ts)
            emit = ts in output_steps
            # Accumulate this step's Fisher contribution across all its blocks
            # before advancing w, so the whole batch B_ts forms one (I − η H_ts)
            # factor (applying blocks one at a time would not compose).
            delta: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(w[name]) for name in layers
            }
            for _s, train_g, train_hashes in iter_gradient_blocks(
                train_fm, [ts], self.args, self.layer_name
            ):
                mat = self._materialize(train_g, device)
                shared = [n for n in layers if n in mat]
                if not shared:
                    continue
                batch = mat[shared[0]].shape[0]
                # D[i, j] = ⟨g(z*_i), w_j⟩ summed over layers  → (B, num_test).
                D = torch.zeros(batch, num_test, device=device)
                for name in shared:
                    D += mat[name] @ w[name].T
                if emit:
                    row_chunks.append((lr * D).detach().to("cpu", torch.float))
                    row_train_ids.extend(train_hashes)
                    row_steps.extend([ts] * batch)
                # Fisher update term: Σ_i D[i, j] g(z*_i)  → (num_test, d).
                for name in shared:
                    delta[name] += D.T @ mat[name]
            for name in layers:
                w[name] -= lr * delta[name]

        scores = (
            torch.cat(row_chunks, dim=0)
            if row_chunks
            else torch.zeros(0, num_test, dtype=torch.float)
        )

        result = AttributionScore(
            scores=scores,
            row_train_ids=row_train_ids,
            row_steps=row_steps,
            test_ids=test_ids,
            algorithm_meta={
                "final_step": final_step,
                "selected_training_steps": sorted(output_steps),
                "propagated_steps": sorted(prop_steps),
                "learning_rate": self.learning_rate,
            },
            algorithm=self.algorithm,
            normalized_grad=False,
            layer_name=self.layer_name,
        )
        result.save(self.args.output_path)
        return result
