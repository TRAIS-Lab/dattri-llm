"""AdamW-influence: unrolled first-order influence through AdamW training.

AdamW-influence (arXiv:2605.18814) linearizes the whole AdamW trajectory
``theta_0 -> theta_T``.  Down-weighting sample ``z`` at the step ``t*`` it was
consumed perturbs the optimizer state by ``Z_push(z) = (theta_dot, m_dot,
v_dot)``; that perturbation is carried to the final parameters by the
product of per-step transition Jacobians, and the score is its effect on a
query's loss at ``theta_T``.  The backward sweep of Algorithm 1 walks the
steps from the last down, scoring every sample as it passes the sample's
step, so the trajectory is swept once whatever the number of samples.

Every use of the propagation is linear, so it can be carried on either side
(the ``propagation`` argument, as for DVEmb):

* ``"train"`` -- the paper's summary ``W = [W_theta | W_m | W_v]``, a
  ``(p, 3p)`` matrix over the ``p`` captured coordinates, test-independent;
  memory ``3 p^2``, so it needs the coordinate mask (a ``"mask"`` capture)
  or small layers.
* ``"test"`` -- ``U = Q W``, the propagation applied to the ``n_test``
  query gradients ``Q`` directly, ``(n_test, 3p)``; matrix-free in ``p``,
  so it runs on every coordinate of the model (no mask), at a cost linear
  in the number of queries.  ``loop_over_test`` re-sweeps the trajectory
  per query block to bound memory to one block.

The gradients come from the trajectory pass -- stored per step, or (with
``args.recompute_gradients``) recomputed from parameter snapshots at sweep
time.  Both use the same recorded optimizer dynamics: the moments before
and after every update on the captured coordinates
(:class:`~dattri_llm.gradient.callbacks.OptimizerStateCallback`), from
which the batch gradient the optimizer actually consumed -- clipping,
scaling and all -- is recovered exactly.

The captured per-sample gradient is the sample's share of the *batch* loss
as trained: ``grad l_z / |B|`` under a mean reduction, ``grad l_z`` under a
sum.  Removing the sample perturbs the batch gradient by exactly that share
(the push), while the GGN of the batch loss in terms of those shares is
``|B| sum_z g_z g_z^T`` for a mean and ``sum_z g_z g_z^T`` for a sum -- the
``loss_reduction`` argument fixes that scale, as in DVEmb.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar

import torch

from dattri_llm.attribution.algorithm.trajectory import (
    TrajectoryAttributor,
    _dense_float,
)
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks.optimizer_state_callback import (
    OptimizerStateCallback,
    assemble_dynamics,
)
from dattri_llm.gradient.snapshots import LazyDynamics, TrajectorySnapshots
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.gradient.streaming import DiskGradientSource, ReplayGradientSource

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import Dataset

    from dattri_llm.attribution.algorithm.trajectory import TrainSource, TrainSourceSpec
    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.callbacks import HookManagerCallback
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import GradientStreamer
    from dattri_llm.task import AttributionTask

_DYNAMICS_FILE = "adamw_dynamics.pt"

# Bytes of per-sample temporaries the query-side sweep forms at once (the
# ``(c, 3p)`` push and the three ``(c, p)`` responses of ``c`` samples): sets
# how many samples of a block are pushed together.
_SWEEP_CHUNK_BYTES = 3 << 30


def _concat_layers(block: Gradient, layers: list[str]) -> torch.Tensor:
    """The block's dense per-sample entries over *layers*, ``(B, p)``."""
    parts = []
    for name in layers:
        value = block.data[name]
        if not isinstance(value, torch.Tensor):
            value = ops.materialize(value, block.layer_types[name])
        parts.append(value.reshape(value.shape[0], -1).float())
    return torch.cat(parts, dim=1)


def _concat_state(
    state: dict[str, tuple], layers: list[str], which: int
) -> torch.Tensor:
    return torch.cat([state[name][which].reshape(-1).float() for name in layers])


class AdamWInfluenceAttributor(TrajectoryAttributor):
    """AdamW-influence attributor.

    Args:
        args: :class:`AttributionArguments`; the trajectory is trained with
            its optimizer settings, exactly as the updating streamer mirrors
            the HF ``Trainer``.
        task: The attribution task; its first checkpoint starts the trajectory.
        projection_kwargs: The projection config the gradients are captured
            with (``"mask"``, the coordinate mask); read off the
            ``hook_config`` of :meth:`cache` when unset.  ``None`` keeps every
            coordinate of the hooked layers.
    """

    algorithm: ClassVar[str] = "AdamWInfluence"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
        projection_kwargs: dict[str, dict] | None = None,
    ) -> None:
        super().__init__(args, task=task)
        self._projection = projection_kwargs
        self._recorder: OptimizerStateCallback | None = None

    # ------------------------------------------------------------------ #
    # Collection                                                           #
    # ------------------------------------------------------------------ #
    def trajectory_callbacks(
        self,
        model: nn.Module,
        streamer: GradientStreamer,
        snapshots: TrajectorySnapshots | None,
    ) -> list[HookManagerCallback]:
        """The moments recorder, on the captured coordinates; to disk when
        recomputing.
        """
        if self._projection is None and self._hook_config is not None:
            self._projection = self._hook_config.projection_kwargs
        self._recorder = OptimizerStateCallback(
            model,
            lambda: streamer.optimizer,
            projection_kwargs=self._projection,
            snapshots=snapshots,
        )
        return [self._recorder]

    def on_trajectory_block(self) -> Callable[[int, Gradient, list[str]], None]:
        """Read the post-update moments once each block's update has run."""
        recorder = self._recorder
        if recorder is None:
            raise RuntimeError("trajectory_callbacks() must run first.")
        return lambda step, _g, _h: recorder.record_post(step)

    def finish_trajectory(
        self,
        train_target: GradientStorageManager | TrajectorySnapshots,
    ) -> None:
        """Write the dynamics beside a gradient store (a snapshot store
        already holds them per step).
        """
        if isinstance(train_target, GradientStorageManager) and self._recorder:
            torch.save(
                dict(self._recorder.dynamics()),
                pathlib.Path(train_target.save_dir) / _DYNAMICS_FILE,
            )

    @staticmethod
    def load_dynamics(
        train: TrainSource,
        dynamics: Mapping[int, dict] | str | pathlib.Path | None,
    ) -> Mapping[int, dict]:
        """The per-step dynamics: as given, from a file, or from the train
        side (the file :meth:`cache` wrote beside a store, or the snapshot
        store's per-step entries, read on access).
        """
        if isinstance(dynamics, Mapping):
            return dynamics
        if dynamics is None:
            if isinstance(train, ReplayGradientSource):
                return LazyDynamics(train.snapshots, assemble_dynamics)
            dynamics = pathlib.Path(train.file_manager.save_dir) / _DYNAMICS_FILE
        return torch.load(dynamics, weights_only=False)

    # ------------------------------------------------------------------ #
    # The sweeps                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _step_terms(
        dyn: dict, layers: list[str], device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, float]:
        """``(g_t, D_t, S_t, common, weight_decay)`` of one step's dynamics."""
        beta1, beta2 = dyn["betas"]
        count, lr, eps = int(dyn["step"]), float(dyn["lr"]), float(dyn["eps"])
        m_pre = _concat_state(dyn["pre"], layers, 0).to(device)
        m_post = _concat_state(dyn["post"], layers, 0).to(device)
        v_post = _concat_state(dyn["post"], layers, 1).to(device)
        # The batch gradient the optimizer consumed, from the first-moment
        # update it produced -- exact whatever clipping or scaling did.
        g_t = (m_post - beta1 * m_pre) / (1.0 - beta1)
        d, s = ops.adam_preconditioner(
            m_post / (1.0 - beta1**count),
            v_post / (1.0 - beta2**count),
            eps=eps,
        )
        common = {"lr": lr, "step": count, "beta1": beta1, "beta2": beta2}
        return g_t, d, s, common, float(dyn["weight_decay"])

    def _step_block(
        self, train: TrainSource, step: int, layers: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """The step's per-sample entries ``(B, p)`` and hashes, over its blocks."""
        parts, ids = [], []
        for _s, block, hashes in train.for_steps([step]):
            parts.append(_concat_layers(block, layers).to(self.args.device))
            ids.extend(hashes)
        if not parts:
            raise ValueError(f"No train gradients recorded at step {step}.")
        return torch.cat(parts, dim=0), ids

    def sweep(
        self,
        train: TrainSource,
        dynamics: Mapping[int, dict],
        test_rep: torch.Tensor,
        *,
        layers: list[str],
        prop_steps: list[int],
        output_steps: set[int],
        loss_reduction: str = "mean",
        verbose: bool = False,
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Algorithm 1 on the training side: carry ``W`` (``(p, 3p)``) from
        the last step down, scoring every sample as it passes.

        Args:
            train: The train side, with per-step random access.
            dynamics: ``{step: {...}}`` from :class:`OptimizerStateCallback`.
            test_rep: ``(n_test, p)`` query gradient entries at ``theta_T``.
            layers: Layer order defining the ``p`` coordinates.
            prop_steps: Steps to propagate through.
            output_steps: Steps whose samples become rows.
            loss_reduction: How the training loss was reduced over each batch,
                ``"mean"`` or ``"sum"``; fixes the scale of the GGN term (see
                the module docstring).
            verbose: Show the per-step progress bar.

        Returns:
            ``(scores, row_train_ids, row_steps)`` with ``scores`` of shape
            ``(rows, n_test)``: the paper's ``-grad(z')^T W Z_push(z)``.
        """
        device = self.args.device
        p = test_rep.shape[1]
        test_rep = test_rep.to(device)
        w_theta = torch.eye(p, device=device)
        w_m = torch.zeros(p, p, device=device)
        w_v = torch.zeros(p, p, device=device)
        rows: list[torch.Tensor] = []
        row_ids: list[str] = []
        row_steps: list[int] = []
        for step in self.steps_bar(prop_steps, "AdamW-influence: sweep (W)", verbose):
            g_z, ids = self._step_block(train, step, layers)
            g_t, d, s, common, weight_decay = self._step_terms(
                dynamics[step], layers, device
            )
            if step in output_steps:
                z = ops.adamw_influence_push(g_z, g_t, d, s, **common)  # (B, 3p)
                influence = (
                    z[:, :p] @ w_theta.T
                    + z[:, p : 2 * p] @ w_m.T
                    + z[:, 2 * p :] @ w_v.T
                )
                rows.append((-(influence @ test_rep.T)).to("cpu"))
                row_ids.extend(ids)
                row_steps.extend([step] * len(ids))
            # W <- W M_t + W R_t H_t on the parameter block, with the GGN
            # H_t = fisher_scale * sum_z g_z g_z^T of the batch loss as trained.
            v = ops.adamw_influence_coupling(
                w_theta, w_m, w_v, g_z, g_t, d, s, **common
            )
            w_theta, w_m, w_v = ops.adamw_influence_transition(
                w_theta, w_m, w_v, d, s, weight_decay=weight_decay, **common
            )
            fisher_scale = float(g_z.shape[0]) if loss_reduction == "mean" else 1.0
            w_theta += fisher_scale * (v.T @ g_z)
        scores = torch.cat(rows, dim=0) if rows else torch.zeros(0, test_rep.shape[0])
        return scores, row_ids, row_steps

    def sweep_queries(
        self,
        train: TrainSource,
        dynamics: Mapping[int, dict],
        queries: torch.Tensor,
        *,
        layers: list[str],
        prop_steps: list[int],
        output_steps: set[int],
        loss_reduction: str = "mean",
        verbose: bool = False,
        block_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Algorithm 1 on the query side: carry ``U = Q W`` (``(n_q, 3p)``)
        for the query gradients ``Q`` instead of ``W`` itself.

        Same arguments and result as :meth:`sweep`, with *queries* the
        ``(n_q, p)`` query gradient entries at ``theta_T`` and *block_dtype*
        the dtype each step's materialized train block is held in.  The
        sweep works layer by layer: ``U`` and the moments are kept per layer,
        and the push and response of a step's samples are formed one layer
        and one chunk of samples at a time (bounded by ``_SWEEP_CHUNK_BYTES``),
        so memory is ``3 n_q p`` plus the step's block plus one chunk.
        """
        device = self.args.device
        n_q = queries.shape[0]
        widths = self._layer_widths(train, prop_steps, layers)
        offsets, start = {}, 0
        for name in layers:
            offsets[name] = (start, start + widths[name])
            start += widths[name]
        if start != queries.shape[1]:
            raise ValueError(
                f"query width {queries.shape[1]} does not match the train side's "
                f"{start} coordinates over {len(layers)} layers."
            )
        queries = queries.to(device).float()
        u_theta = {n: queries[:, offsets[n][0] : offsets[n][1]].clone() for n in layers}
        u_m = {n: torch.zeros_like(u_theta[n]) for n in layers}
        u_v = {n: torch.zeros_like(u_theta[n]) for n in layers}
        del queries
        rows: list[torch.Tensor] = []
        row_ids: list[str] = []
        row_steps: list[int] = []
        for step in self.steps_bar(prop_steps, "AdamW-influence: sweep (U)", verbose):
            blocks, ids = self._step_blocks(train, step, layers, block_dtype)
            batch = ids and blocks[layers[0]].shape[0]
            dyn = dynamics[step]
            emit = step in output_steps
            scores = torch.zeros(batch, n_q, device=device)
            coup = torch.zeros(batch, n_q, device=device)
            terms = {}
            for name in layers:
                g_t, d, s, common, weight_decay = self._layer_terms(dyn, name, device)
                terms[name] = (g_t, d, s, common, weight_decay)
                w = widths[name]
                chunk = max(1, _SWEEP_CHUNK_BYTES // (6 * w * 4))
                for c0 in range(0, batch, chunk):
                    g_c = blocks[name][c0 : c0 + chunk].float()
                    if emit:
                        z = ops.adamw_influence_push(g_c, g_t, d, s, **common)
                        scores[c0 : c0 + chunk] -= (
                            z[:, :w] @ u_theta[name].T
                            + z[:, w : 2 * w] @ u_m[name].T
                            + z[:, 2 * w :] @ u_v[name].T
                        )
                        del z
                    coup[c0 : c0 + chunk] += ops.adamw_influence_coupling(
                        u_theta[name], u_m[name], u_v[name], g_c, g_t, d, s, **common
                    )
                    del g_c
            if emit:
                rows.append(scores.to("cpu"))
                row_ids.extend(ids)
                row_steps.extend([step] * batch)
            fisher_scale = float(batch) if loss_reduction == "mean" else 1.0
            for name in layers:
                g_t, d, s, common, weight_decay = terms[name]
                # (U R_t g_z)^T g_z summed over the batch, then the transition.
                fisher_u = coup.T @ blocks[name].float()
                u_theta[name], u_m[name], u_v[name] = ops.adamw_influence_transition(
                    u_theta[name],
                    u_m[name],
                    u_v[name],
                    d,
                    s,
                    weight_decay=weight_decay,
                    **common,
                )
                u_theta[name] += fisher_scale * fisher_u
                del fisher_u
            del blocks, terms, coup, scores
        scores_all = torch.cat(rows, dim=0) if rows else torch.zeros(0, n_q)
        return scores_all, row_ids, row_steps

    def _layer_widths(
        self, train: TrainSource, prop_steps: list[int], layers: list[str]
    ) -> dict[str, int]:
        """Flat width of every layer, peeked from one train block."""
        for _s, block, _h in train.for_steps([max(prop_steps)]):
            mat = _dense_float(block.to(self.args.device)).data
            return {n: mat[n].shape[1] for n in layers}
        raise ValueError(f"No train gradients recorded at step {max(prop_steps)}.")

    def _step_blocks(
        self,
        train: TrainSource,
        step: int,
        layers: list[str],
        dtype: torch.dtype,
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """The step's per-layer ``(B, w)`` entries in *dtype*, and its hashes."""
        parts: dict[str, list[torch.Tensor]] = {n: [] for n in layers}
        ids: list[str] = []
        for _s, block, hashes in train.for_steps([step]):
            mat = _dense_float(block.to(self.args.device), dtype).data
            for name in layers:
                parts[name].append(mat[name])
            ids.extend(hashes)
        if not ids:
            raise ValueError(f"No train gradients recorded at step {step}.")
        return {
            n: torch.cat(v, dim=0) if len(v) > 1 else v[0] for n, v in parts.items()
        }, ids

    @staticmethod
    def _layer_terms(
        dyn: dict, name: str, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, float]:
        """``(g_t, D_t, S_t, common, weight_decay)`` of one layer at one step."""
        beta1, beta2 = dyn["betas"]
        count, lr, eps = int(dyn["step"]), float(dyn["lr"]), float(dyn["eps"])
        m_pre = dyn["pre"][name][0].reshape(-1).float().to(device)
        m_post = dyn["post"][name][0].reshape(-1).float().to(device)
        v_post = dyn["post"][name][1].reshape(-1).float().to(device)
        g_t = (m_post - beta1 * m_pre) / (1.0 - beta1)
        d, s = ops.adam_preconditioner(
            m_post / (1.0 - beta1**count), v_post / (1.0 - beta2**count), eps=eps
        )
        common = {"lr": lr, "step": count, "beta1": beta1, "beta2": beta2}
        return g_t, d, s, common, float(dyn["weight_decay"])

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
        **kwargs: object,
    ) -> AttributionScore:
        """Run the trajectory, cache it under ``args.output_dir``, and score."""
        ((train_dir, test_dir),) = self.cache(
            train_dataset, test_dataset, hook_config=hook_config
        )
        return self.attribute_from_cache(
            train_dir, test_dir, hook_config=hook_config, verbose=verbose, **kwargs
        )

    def attribute_from_cache(
        self,
        train_source: TrainSourceSpec,
        test_source: str | pathlib.Path | GradientStorageManager | DiskGradientSource,
        *,
        dynamics: Mapping[int, dict] | str | pathlib.Path | None = None,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        loss_reduction: str = "mean",
        propagation: str = "train",
        loop_over_test: bool = False,
        final_step: int | None = None,
        hook_config: HookManagerConfig | None = None,
        block_dtype: torch.dtype = torch.float32,
        **kwargs: object,
    ) -> AttributionScore:
        """Sweep a cached trajectory.

        Args:
            train_source: The train side: per-step train gradients (a store
                directory, open store or disk source), or the snapshot store
                (directory or :class:`TrajectorySnapshots`) to recompute
                them from -- which needs the ``task``.
            test_source: Query gradients at the final model.
            dynamics: The per-step optimizer dynamics, or a path to the file
                :meth:`cache` wrote; defaults to that file beside a gradient
                store, or to a snapshot store's own entries.
            selected_training_steps: Restrict the output rows to these steps.
                The sweep still propagates through every later step.
            layer_name: Restrict to this subset of the stored layers.
            verbose: Show progress bars.
            loss_reduction: How the training loss was reduced over each
                batch, ``"mean"`` (default) or ``"sum"`` (see :meth:`sweep`).
            propagation: Which side carries the propagation -- ``"train"``
                (default; the ``(p, 3p)`` summary, for masked captures) or
                ``"test"`` (through the query gradients; matrix-free in
                ``p``, memory ``3 n_test p``).
            loop_over_test: ``propagation="test"`` only: sweep once per
                query block instead of holding every query, bounding memory
                to one block at the cost of re-reading (or recomputing) the
                train side per block.  Identical scores.
            final_step: The step index of the final model the query
                gradients were taken at; ``None`` uses one past the last
                stored step.
            hook_config: When recomputing, the capture configuration the
                trajectory was collected with (that of this attributor's
                last :meth:`cache` when unset).
            block_dtype: ``propagation="test"`` only: the dtype each step's
                materialized train block is held in (the push and coupling
                are formed in float32 per layer and chunk).  A narrower dtype
                halves the resident block of an unprojected sweep.
            **kwargs: Recorded in the score's metadata.
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}."
            )
        self.validate_propagation(propagation, loop_over_test)
        train = self.open_train_source(
            train_source,
            layer_name=layer_name,
            hook_config=hook_config,
            verbose=verbose,
        )
        try:
            test_store = self.resolve_store(test_source)
            test = self.load_test_rep(
                test_store, layer_name=layer_name, verbose=verbose
            )
            dyn = self.load_dynamics(train, dynamics)
            prop_steps, output_steps, final_step = self.resolve_steps(
                self.available_steps(train), selected_training_steps, final_step
            )
            missing = [s for s in prop_steps if s not in dyn]
            if missing:
                raise ValueError(
                    f"no optimizer dynamics recorded for steps {missing[:5]}..."
                )
            layers: list[str] | None = None
            for _step, block, _hashes in test:
                layers = sorted(block.data)
                break
            if layers is None:
                raise ValueError("the test source yielded no blocks.")
            sweep_kw = {
                "layers": layers,
                "prop_steps": prop_steps,
                "output_steps": output_steps,
                "loss_reduction": loss_reduction,
                "verbose": verbose,
            }
            if propagation == "test":
                sweep_kw["block_dtype"] = block_dtype
            if propagation == "train" or not loop_over_test:
                test_ids, parts = [], []
                for _step, block, hashes in test:
                    parts.append(_concat_layers(block, layers))
                    test_ids.extend(hashes)
                test_rep = torch.cat(parts, dim=0)
                run = self.sweep if propagation == "train" else self.sweep_queries
                scores, row_ids, row_steps = run(train, dyn, test_rep, **sweep_kw)
            else:
                test_ids, test_index = self.test_column_order(test)
                scores = None
                row_ids, row_steps = [], []
                for _step, block, hashes in test:
                    block_scores, rids, rsteps = self.sweep_queries(
                        train, dyn, _concat_layers(block, layers), **sweep_kw
                    )
                    if scores is None:
                        scores = torch.zeros(block_scores.shape[0], len(test_ids))
                        row_ids, row_steps = rids, rsteps
                    scores[:, [test_index[h] for h in hashes]] = block_scores
                if scores is None:
                    scores = torch.zeros(0, len(test_ids))
            meta = {
                "final_step": final_step,
                "selected_training_steps": sorted(output_steps),
                "propagated_steps": sorted(prop_steps),
                "loss_reduction": loss_reduction,
                "propagation": propagation,
                "loop_over_test": loop_over_test,
                "block_dtype": str(block_dtype),
                **self.source_meta(train, test_store),
                **kwargs,
            }
        finally:
            self.close_source(train)
        return self.build_score(
            scores, row_ids, row_steps, test_ids, algorithm_meta=meta, layer_name=layers
        )
