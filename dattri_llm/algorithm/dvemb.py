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

    H_k ≈ (1/c) Σ_{z ∈ B_k} ĝ(θ_k, z) ĝ(θ_k, z)ᵀ                            (2)

where ``ĝ`` is the *recorded* per-sample gradient and ``c`` its per-sample loss
weight: the empirical Fisher must use the **true** per-sample gradients, so when
the gradients were recorded under a **mean** loss (``ĝ = ∇ℓ / B``, ``c = 1/B``)
the sum of recorded outer products is rescaled by the step's batch size ``B``,
while under a **sum** loss (``ĝ = ∇ℓ``, ``c = 1``) it is used as-is — see the
``loss_reduction`` argument of :meth:`attribute`.

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
from tqdm.auto import tqdm

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

    def _propagate_and_score(
        self,
        w: Dict[str, torch.Tensor],
        n_cols: int,
        layers: List[str],
        train_fm: GradientFileManager,
        prop_steps: List[int],
        output_steps: set,
        device: torch.device,
        loss_reduction: str,
        verbose: bool,
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """One latest→earliest sweep for the ``n_cols`` columns held in ``w``.

        The step is the outer loop, so each train block is loaded/materialised
        once **per call**; ``w[name]`` (shape ``(n_cols, d)``) is scored against
        the step's train gradients and then advanced in place by that step's
        Fisher factor.  Because every test column propagates independently,
        scoring a subset of columns gives identical values to scoring them all —
        this is what makes the ``loop_over_test`` column-blocking exact.

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
            lr = self._lr(ts)
            emit = ts in output_steps
            # Accumulate this step's Fisher contribution across all its blocks
            # before advancing w, so the whole batch B_ts forms one (I − η H_ts)
            # factor.  ``n_t`` counts the step's recorded samples for B_ts.
            delta: Dict[str, torch.Tensor] = {
                name: torch.zeros_like(w[name]) for name in layers
            }
            n_t = 0
            for _s, train_g, train_hashes in iter_gradient_blocks(
                train_fm, [ts], self.args, self.layer_name
            ):
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
        loss_reduction: str = "mean",
        verbose: bool = False,
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
            loop_over_test: Memory/disk trade-off for the per-test-column
                embedding ``w`` (shape ``(num_test, d)``), which is
                back-propagated through the steps.  Because every test column
                propagates independently, ``False`` (default) uses the test
                columns as the **inner** loop and the steps as the outer loop:
                one dense embedding is held and the training gradients are
                streamed exactly **once** (peak memory: full ``w`` + one train
                block; fastest).  ``True`` uses the test blocks as the **outer**
                loop to save space: each test block gets its own sweep, holding
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

        Returns:
            An :class:`AttributionScore`; also persisted to ``args.output_dir``.

        Raises:
            ValueError: If a gradients dir is missing, ``loss_reduction`` is not
                ``"mean"``/``"sum"``, ``selected_training_steps`` matches no
                available step, or no training step satisfies
                ``step < final_step``.
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}."
            )
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

        # The test embedding w starts at the final-model test gradients and is
        # back-propagated through the steps.  Each test column evolves
        # independently, so loop_over_test trades memory for disk reads.
        if not loop_over_test:
            # ---- step outer / test inner: one dense embedding, train read once.
            # Materialise every test gradient into one (num_test, d) embedding and
            # sweep the train gradients a single time (peak: full w + one train
            # block).  Fastest; the default.
            test_ids: List[str] = []
            test_index: Dict[str, int] = {}
            pending: List[Tuple[Dict[str, torch.Tensor], List[int]]] = []
            for _step, test_g, test_hashes in iter_gradient_blocks(
                test_fm, test_steps, self.args, self.layer_name,
                desc="DVEmb: loading test",
                verbose=verbose,
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
                    # Duplicate-hash columns collapse via index_copy_ (last wins).
                    w[name].index_copy_(0, idx, mat[name])
            del pending
            scores, row_train_ids, row_steps = self._propagate_and_score(
                w, num_test, layers, train_fm, prop_steps, output_steps,
                device, loss_reduction, verbose,
            )
        else:
            # ---- test outer: one block's embedding resident, train re-streamed.
            # Bounds peak memory to a single test block's embedding by giving each
            # test block its own full sweep (re-reading the train gradients once
            # per block).  Use when the full test embedding does not fit.
            # Pass 1 fixes the column order from hashes alone (no materialise).
            test_ids = []
            test_index = {}
            for _step, _tg, test_hashes in iter_gradient_blocks(
                test_fm, test_steps, self.args, self.layer_name,
                desc="DVEmb: indexing test",
                verbose=verbose,
            ):
                for h in test_hashes:
                    if h not in test_index:
                        test_index[h] = len(test_ids)
                        test_ids.append(h)
            num_test = len(test_ids)
            # Pass 2: one test block at a time — seed w, sweep, scatter columns.
            scores = None
            row_train_ids = []
            row_steps = []
            for _step, test_g, test_hashes in iter_gradient_blocks(
                test_fm, test_steps, self.args, self.layer_name,
                desc="DVEmb: scoring test blocks",
                verbose=verbose,
            ):
                w_block = self._materialize(test_g, device)  # (B_block, d) per layer
                layers = list(w_block.keys())
                block_cols = [test_index[h] for h in test_hashes]
                block_scores, rtids, rsteps = self._propagate_and_score(
                    w_block, len(block_cols), layers, train_fm, prop_steps,
                    output_steps, device, loss_reduction, verbose,
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
            algorithm_meta={
                "final_step": final_step,
                "selected_training_steps": sorted(output_steps),
                "propagated_steps": sorted(prop_steps),
                "learning_rate": self.learning_rate,
                "loss_reduction": loss_reduction,
            },
            algorithm=self.algorithm,
            normalized_grad=False,
            layer_name=self.layer_name,
        )
        result.save(self.args.output_path)
        return result
