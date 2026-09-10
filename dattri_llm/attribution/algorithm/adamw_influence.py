"""AdamW-influence: unrolled first-order influence through AdamW training.

AdamW-influence (arXiv:2605.18814) linearizes the whole AdamW trajectory
``theta_0 -> theta_T``.  Down-weighting sample ``z`` at the step ``t*`` it was
consumed perturbs the optimizer state by ``Z_push(z) = (theta_dot, m_dot,
v_dot)``; that perturbation is carried to the final parameters by the
product of per-step transition Jacobians, and the score is its effect on a
query's loss at ``theta_T``.  The backward sweep of Algorithm 1 accumulates
the propagation ``W`` from the last step down, emitting every sample's
``W Z_push(z)`` as it passes the sample's step, so the trajectory is swept
once whatever the number of samples.

Everything -- gradients, moments, the ``(p, 3p)`` summary ``W`` and the GGN
Hessian ``H_t ~ (1/|B|) sum_z grad l_z grad l_z^T`` -- lives on the coordinate set
``S`` the gradients were captured on: a ``"subset_materialized"`` subset of
the hooked layers (the paper's random mask), or every coordinate of small
layers.  Memory is three ``|S| x |S|`` blocks.

What the trajectory pass records, per step: the raw per-sample gradient
entries (the train store), and the optimizer's moments on ``S`` both before
and after the update (:class:`~dattri_llm.gradient.callbacks.OptimizerStateCallback`).
The batch gradient the optimizer actually consumed -- clipping, scaling and
all -- is recovered exactly from the two moment snapshots.

The captured per-sample gradient is the sample's share of the *batch* loss
as trained: ``grad l_z / |B|`` under a mean reduction, ``grad l_z`` under a
sum.  Removing the sample perturbs the batch gradient by exactly that share
(the push), while the GGN of the batch loss in terms of those shares is
``|B| sum_z g_z g_z^T`` for a mean and ``sum_z g_z g_z^T`` for a sum -- the
``loss_reduction`` argument fixes that scale, as in DVEmb.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import TYPE_CHECKING, ClassVar

import torch

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks.optimizer_state_callback import (
    OptimizerStateCallback,
)
from dattri_llm.gradient.storage_manager import GradientStorageManager

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.gradient import Gradient
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import DiskGradientSource

_DYNAMICS_FILE = "adamw_dynamics.pt"


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


class AdamWInfluenceAttributor(BaseInnerProductAttributor):
    """AdamW-influence attributor.

    Args:
        args: :class:`AttributionArguments`; the trajectory is trained with
            its optimizer settings, exactly as the updating streamer mirrors
            the HF ``Trainer``.
        task: The attribution task; its first checkpoint starts the trajectory.
        projection: The projection config the gradients are captured with
            (``"subset_materialized"``, the coordinate mask); read off the
            ``hook_config`` of :meth:`cache` when unset.  ``None`` keeps every
            coordinate of the hooked layers.
    """

    algorithm: ClassVar[str] = "AdamWInfluence"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
        projection: dict[str, dict] | None = None,
    ) -> None:
        super().__init__(args, task=task)
        self._projection = projection

    # ------------------------------------------------------------------ #
    # Collection                                                           #
    # ------------------------------------------------------------------ #
    def checkpoints(self) -> list[int]:
        """The trajectory regenerates from the task's first checkpoint."""
        n = self.num_checkpoints()
        if n > 1:
            warnings.warn(
                f"AdamW-influence regenerates the trajectory from checkpoint 0; "
                f"the other {n - 1} checkpoint(s) are ignored.",
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
    ) -> list[tuple[str, str]]:
        """Run the AdamW trajectory live and cache what the sweep needs.

        Writes the per-step train gradients to ``<cache_dir>/train_grads``,
        the query gradients at the final model to ``<cache_dir>/test_grads``,
        and the per-step optimizer dynamics beside the train store.
        """
        self.require_task("cache")
        cache_dir = cache_dir if cache_dir is not None else self.args.output_dir
        train_dir = str(pathlib.Path(cache_dir) / "train_grads")
        test_dir = str(pathlib.Path(cache_dir) / "test_grads")
        dynamics = self.collect_trajectory(
            GradientStorageManager(train_dir),
            GradientStorageManager(test_dir),
            train_dataset,
            test_dataset,
            hook_config=hook_config,
        )
        torch.save(dynamics, pathlib.Path(train_dir) / _DYNAMICS_FILE)
        return [(train_dir, test_dir)]

    def collect_trajectory(
        self,
        train_store: GradientStorageManager,
        test_store: GradientStorageManager,
        train_dataset: Dataset,
        test_dataset: Dataset,
        *,
        hook_config: HookManagerConfig | None = None,
    ) -> dict[int, dict]:
        """Drive the trajectory into *train_store* (with the optimizer's
        moments snapshotted around every update) and the final-model query
        pass into *test_store*; return the per-step dynamics.
        """
        self.checkpoints()
        model = self.load_checkpoint(0)
        if hook_config is not None and self._projection is None:
            self._projection = hook_config.projection
        streamer = self.generate_train_rep(
            train_dataset,
            enable_update=True,
            hook_config=hook_config,
        )
        recorder = OptimizerStateCallback(
            model, lambda: streamer.optimizer, projection=self._projection
        )
        streamer.hook_manager.add_callback(recorder)
        self.collect_gradients(
            streamer,
            train_store,
            on_block=lambda step, _g, _h: recorder.record_post(step),
        )
        self.collect_gradients(
            self.generate_test_rep(test_dataset, hook_config=hook_config),
            test_store,
        )
        return recorder.dynamics()

    # ------------------------------------------------------------------ #
    # The sweep                                                            #
    # ------------------------------------------------------------------ #
    def sweep(
        self,
        train_blocks: Iterable[tuple[int, Gradient, list[str]]],
        dynamics: dict[int, dict],
        test_rep: torch.Tensor,
        *,
        layers: list[str],
        loss_reduction: str = "mean",
    ) -> tuple[torch.Tensor, list[str], list[int]]:
        """Algorithm 1: backward over the steps, scoring every sample as it passes.

        Args:
            train_blocks: ``(step, block, hashes)`` for every trajectory step.
            dynamics: ``{step: {...}}`` from :class:`OptimizerStateCallback`.
            test_rep: ``(n_test, p)`` query gradient entries at ``theta_T``.
            layers: Layer order defining the ``p`` coordinates.
            loss_reduction: How the training loss was reduced over each batch,
                ``"mean"`` or ``"sum"``; fixes the scale of the GGN term (see
                the module docstring).

        Returns:
            ``(scores, row_train_ids, row_steps)`` with ``scores`` of shape
            ``(rows, n_test)``: the paper's ``-grad(z')^T W Z_push(z)``.
        """
        if loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"loss_reduction must be 'mean' or 'sum', got {loss_reduction!r}."
            )
        device = self.args.device
        blocks: dict[int, tuple[torch.Tensor, list[str]]] = {}
        for step, block, hashes in train_blocks:
            if step in blocks:
                raise ValueError(
                    f"step {step} appears more than once; the sweep needs one "
                    "block per optimizer step (collect with offload_interval=1).",
                )
            blocks[step] = (_concat_layers(block, layers).to(device), list(hashes))
        missing = sorted(set(blocks) - set(dynamics))
        if missing:
            raise ValueError(
                f"no optimizer dynamics recorded for steps {missing[:5]}..."
            )
        p = test_rep.shape[1]
        test_rep = test_rep.to(device)
        w_theta = torch.eye(p, device=device)
        w_m = torch.zeros(p, p, device=device)
        w_v = torch.zeros(p, p, device=device)
        rows: list[torch.Tensor] = []
        row_ids: list[str] = []
        row_steps: list[int] = []
        for step in sorted(blocks, reverse=True):
            g_z, ids = blocks[step]
            dyn = dynamics[step]
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
            z = ops.adamw_influence_push(g_z, g_t, d, s, **common)  # (B, 3p)
            influence = (
                z[:, :p] @ w_theta.T + z[:, p : 2 * p] @ w_m.T + z[:, 2 * p :] @ w_v.T
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
                w_theta,
                w_m,
                w_v,
                d,
                s,
                weight_decay=float(dyn["weight_decay"]),
                **common,
            )
            fisher_scale = float(g_z.shape[0]) if loss_reduction == "mean" else 1.0
            w_theta += fisher_scale * (v.T @ g_z)
        scores = torch.cat(rows, dim=0) if rows else torch.zeros(0, test_rep.shape[0])
        return scores, row_ids, row_steps

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
        return self.attribute_from_cache(train_dir, test_dir, verbose=verbose, **kwargs)

    def attribute_from_cache(
        self,
        train_source: str | Path | GradientStorageManager | DiskGradientSource,
        test_source: str | Path | GradientStorageManager | DiskGradientSource,
        *,
        dynamics: dict[int, dict] | str | None = None,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        loss_reduction: str = "mean",
        **kwargs: object,
    ) -> AttributionScore:
        """Sweep a cached trajectory.

        Args:
            train_source: Per-step train gradients (one block per step).
            test_source: Query gradients at the final model.
            dynamics: The per-step optimizer dynamics, or a path to the file
                :meth:`cache` wrote; defaults to that file beside the train
                store.
            selected_training_steps: Restrict the output rows to these steps.
                The sweep still propagates through every later step.
            layer_name: Restrict to this subset of the stored layers.
            verbose: Show progress bars.
            loss_reduction: How the training loss was reduced over each
                batch, ``"mean"`` (default) or ``"sum"`` (see :meth:`sweep`).
            **kwargs: Recorded in the score's metadata.
        """
        train_store = self.resolve_store(train_source)
        test_store = self.resolve_store(test_source)
        if dynamics is None:
            dynamics = pathlib.Path(train_store.save_dir) / _DYNAMICS_FILE
        if not isinstance(dynamics, dict):
            dynamics = torch.load(dynamics, weights_only=False)
        train = self.load_train_rep(train_store, layer_name=layer_name, verbose=verbose)
        test = self.load_test_rep(test_store, layer_name=layer_name, verbose=verbose)

        test_ids: list[str] = []
        test_parts: list[torch.Tensor] = []
        layers: list[str] | None = None
        for _step, block, hashes in test:
            if layers is None:
                layers = sorted(block.data)
            test_parts.append(_concat_layers(block, layers))
            test_ids.extend(hashes)
        if layers is None:
            raise ValueError("the test source yielded no blocks.")
        test_rep = torch.cat(test_parts, dim=0)

        scores, row_ids, row_steps = self.sweep(
            train, dynamics, test_rep, layers=layers, loss_reduction=loss_reduction
        )
        if selected_training_steps is not None:
            keep = set(selected_training_steps)
            mask = [i for i, s in enumerate(row_steps) if s in keep]
            scores = scores[mask]
            row_ids = [row_ids[i] for i in mask]
            row_steps = [row_steps[i] for i in mask]
        return self.build_score(
            scores,
            row_ids,
            row_steps,
            test_ids,
            algorithm_meta={
                "selected_training_steps": (
                    None if selected_training_steps is None else sorted(keep)
                ),
                "loss_reduction": loss_reduction,
                **self.stores_meta(train_store, test_store),
                **kwargs,
            },
            layer_name=layers,
        )
