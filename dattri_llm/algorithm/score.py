"""Structured attribution scores for the on-disk gradient workflow.

* **Trajectory-aware rows.**  TracIn-style scores are a sum of per-step
  ensemble terms ``Σ_k weight_k · ⟨g_train^k, g_test^k⟩``.  We keep those
  terms *un-summed*: every row of :attr:`scores` is a single
  ``(train_hash, step)`` pair.  Summing a training sample's rows over steps
  recovers the trajectory-*agnostic* score, so the
  finer representation loses no information.  The test side stays
  trajectory-agnostic: one column per test sample.

* **Hash identifiers.**  Rows and columns are keyed by the content hash of a
  sample's inputs (:func:`~dattri_llm.gradient.utils.hash_sample`), not by a
  dataset index.  Hashes are stable under reshuffling and independent of the
  dataset object, so a user holding only an input hash can recover the
  relevant rows/columns without knowing dataset order or which steps a sample
  appeared at.

Two lookup maps, rebuilt from the stored lists on :meth:`load`, make this
queryable:

* :attr:`train_index` — ``train_hash -> [(step, row_idx), ...]``
* :attr:`test_index`  — ``test_hash -> col_idx``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


@dataclass
class AttributionScore:
    """Trajectory-aware attribution scores keyed by input hash.

    Args:
        scores: Float tensor of shape ``(num_rows, num_test)``.  Row ``r``
            holds ``⟨g_train^step, g_test⟩`` for training sample
            ``row_train_ids[r]`` at step ``row_steps[r]``, against every test
            sample.
        row_train_ids: Length-``num_rows`` list giving the training-sample
            hash for each row.
        row_steps: Length-``num_rows`` list giving the training step for each
            row.  ``(row_train_ids[r], row_steps[r])`` is unique across rows.
        test_ids: Length-``num_test`` list giving the test-sample hash for each
            column.
        algorithm: The specific attribution algorithm used.
        algorithm_meta: Algorithm-specific metadata (e.g. ``"damping"`` for the
            K-FAC family); empty for TracIn/GradCos.
        normalized_grad: Whether per-sample gradients were L2-normalised
            (cosine / GradCos) before the inner product.
        layer_name: Layers the inner product was restricted to, or ``None``.
    """

    scores: torch.Tensor
    row_train_ids: List[str]
    row_steps: List[int]
    test_ids: List[str]

    algorithm: str
    algorithm_meta: Dict[str, Any]
    normalized_grad: bool
    layer_name: Optional[List[str]]

    # Rebuilt from the lists above; never persisted directly.
    train_index: Dict[str, List[Tuple[int, int]]] = field(init=False, repr=False)
    test_index: Dict[str, int] = field(init=False, repr=False)

    _SCORES_FILE = "scores.pt"
    _META_FILE = "metadata.json"

    def __post_init__(self) -> None:
        n_rows = self.scores.shape[0]
        if not (len(self.row_train_ids) == len(self.row_steps) == n_rows):
            raise ValueError(
                f"row_train_ids ({len(self.row_train_ids)}) and row_steps "
                f"({len(self.row_steps)}) must both equal num_rows ({n_rows})."
            )
        if len(self.test_ids) != self.scores.shape[1]:
            raise ValueError(
                f"test_ids ({len(self.test_ids)}) must equal num_test "
                f"({self.scores.shape[1]})."
            )
        self._build_indices()

    def _build_indices(self) -> None:
        train_index: Dict[str, List[Tuple[int, int]]] = {}
        for row_idx, (h, step) in enumerate(zip(self.row_train_ids, self.row_steps)):
            train_index.setdefault(h, []).append((step, row_idx))
        for entries in train_index.values():
            entries.sort()  # by (step, row_idx)
        self.train_index = train_index
        self.test_index = {h: col for col, h in enumerate(self.test_ids)}

    # ------------------------------------------------------------------ #
    # Ordering helpers                                                    #
    # ------------------------------------------------------------------ #

    @property
    def train_ids(self) -> List[str]:
        """Distinct training hashes, in first-occurrence (disk) order."""
        seen: Dict[str, None] = {}
        for h in self.row_train_ids:
            seen.setdefault(h, None)
        return list(seen)

    @property
    def num_rows(self) -> int:
        return self.scores.shape[0]

    @property
    def num_train(self) -> int:
        """Number of distinct training samples (collapsing steps)."""
        return len(self.train_index)

    @property
    def num_test(self) -> int:
        return self.scores.shape[1]

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def trajectory_aware(self, train_hash: str) -> Tuple[List[int], torch.Tensor]:
        """Per-step scores for one training sample against every test sample.

        Args:
            train_hash: The training sample's input hash.

        Returns:
            ``(steps, matrix)`` where ``steps`` is the ascending list of steps
            this sample appears at and ``matrix`` has shape
            ``(len(steps), num_test)``.

        Raises:
            KeyError: If ``train_hash`` is not present.
        """
        entries = self._require_train(train_hash)
        steps = [step for step, _ in entries]
        rows = self.scores[[row_idx for _, row_idx in entries]]
        return steps, rows

    def trajectory_agnostic(self, train_hash: str) -> torch.Tensor:
        """Step-summed score row for one training sample.

        Args:
            train_hash: The training sample's input hash.

        Returns:
            Tensor of shape ``(num_test,)`` — the classic TracIn score for this
            sample, summed over all of its steps.
        """
        _, rows = self.trajectory_aware(train_hash)
        return rows.sum(dim=0)

    def agnostic_matrix(self) -> Tuple[List[str], torch.Tensor]:
        """The accumulated ``(num_train, num_test)`` matrix.

        Returns:
            ``(train_ids, matrix)`` where ``train_ids`` are the distinct
            training hashes in disk order and ``matrix`` has shape
            ``(num_train, num_test)``, each row summed over that sample's steps.
        """
        ids = self.train_ids
        out = self.scores.new_zeros(len(ids), self.num_test)
        for i, h in enumerate(ids):
            for _, row_idx in self.train_index[h]:
                out[i] += self.scores[row_idx]
        return ids, out

    def column(self, test_hash: str) -> torch.Tensor:
        """All trajectory-aware scores for one test sample.

        Args:
            test_hash: The test sample's input hash.

        Returns:
            Tensor of shape ``(num_rows,)`` — the test sample's column over
            every ``(train_hash, step)`` row.

        Raises:
            KeyError: If ``test_hash`` is not present.
        """
        if test_hash not in self.test_index:
            raise KeyError(f"test hash {test_hash[:16]}… not in scores.")
        return self.scores[:, self.test_index[test_hash]]

    def step_matrix(self, step: int) -> Tuple[List[str], torch.Tensor]:
        """The ``(num_train, num_test)`` score matrix for a single step.

        This is the per-step view of the attribution: the ensemble term
        ``weight_step · ⟨g_train^step, g_test^step⟩`` for every train/test pair
        present at ``step``.  Summing :meth:`step_matrix` over all recorded
        steps reproduces :meth:`agnostic_matrix`.

        Args:
            step: The step to extract.

        Returns:
            ``(train_ids, matrix)`` where ``train_ids`` are the training hashes
            present at ``step`` in disk order and ``matrix`` has shape
            ``(len(train_ids), num_test)``.

        Raises:
            KeyError: If no rows were recorded for ``step``.
        """
        rows = [i for i, s in enumerate(self.row_steps) if s == step]
        if not rows:
            raise KeyError(f"step {step} not present in scores.")
        train_ids = [self.row_train_ids[i] for i in rows]
        return train_ids, self.scores[rows]

    def step_matrices(self) -> "Dict[int, Tuple[List[str], torch.Tensor]]":
        """Every step's score matrix, keyed by step.

        Convenience wrapper over :meth:`step_matrix` that returns the per-step
        view for *all* steps at once — the natural way to present per-step
        attribution scores to a user.

        Returns:
            Ordered mapping ``step -> (train_ids, matrix)`` for each step in
            recorded steps (ascending), where ``matrix`` has shape
            ``(num_train_at_step, num_test)``.
        """
        present = sorted(set(self.row_steps))
        return {step: self.step_matrix(step) for step in present}

    def score_at(self, train_hash: str, step: int, test_hash: str) -> torch.Tensor:
        """The single attribution score for one ``(train, step, test)`` triple.

        Args:
            train_hash: The training sample's input hash.
            step: The training step.
            test_hash: The test sample's input hash.

        Returns:
            A 0-dim tensor: ``weight_step · ⟨g_train^step, g_test^step⟩`` (or its
            cosine for GradCos).

        Raises:
            KeyError: If the training hash is absent, the test hash is absent,
                or the training sample has no row at ``step``.
        """
        col = self._require_test(test_hash)
        for s, row_idx in self._require_train(train_hash):
            if s == step:
                return self.scores[row_idx, col]
        known = sorted(s for s, _ in self.train_index[train_hash])
        raise KeyError(
            f"train sample {train_hash[:16]}… has no row at step {step}; "
            f"known steps: {known}"
        )

    def query(
        self,
        train_hashes: Optional[Sequence[str]] = None,
        test_hashes: Optional[Sequence[str]] = None,
        trajectory: str = "agnostic",
    ) -> torch.Tensor:
        """Recover the submatrix for arbitrary train/test queries.

        Args:
            train_hashes: Training hashes to include as rows.  ``None`` selects
                all training samples in disk order.
            test_hashes: Test hashes to include as columns.  ``None`` selects
                all test samples in column order.
            trajectory: ``"agnostic"`` (default) returns one step-summed row per
                training sample, shape ``(len(train_hashes), len(test_hashes))``.
                ``"aware"`` returns one row per ``(train_hash, step)`` pair,
                shape ``(num_selected_rows, len(test_hashes))``.

        Returns:
            The requested submatrix.

        Raises:
            ValueError: If ``trajectory`` is not ``"aware"`` or ``"agnostic"``.
            KeyError: If any requested hash is absent.
        """
        if trajectory not in ("aware", "agnostic"):
            raise ValueError(
                f"trajectory must be 'aware' or 'agnostic', got {trajectory!r}."
            )
        train_hashes = list(train_hashes) if train_hashes is not None else self.train_ids
        if test_hashes is None:
            cols = list(range(self.num_test))
        else:
            cols = [self._require_test(h) for h in test_hashes]

        if trajectory == "agnostic":
            rows = torch.stack(
                [self.trajectory_agnostic(h) for h in train_hashes], dim=0
            )
            return rows[:, cols]

        row_idxs: List[int] = []
        for h in train_hashes:
            row_idxs.extend(row_idx for _, row_idx in self._require_train(h))
        return self.scores[row_idxs][:, cols]

    def _require_train(self, train_hash: str) -> List[Tuple[int, int]]:
        if train_hash not in self.train_index:
            raise KeyError(f"train hash {train_hash[:16]}… not in scores.")
        return self.train_index[train_hash]

    def _require_test(self, test_hash: str) -> int:
        if test_hash not in self.test_index:
            raise KeyError(f"test hash {test_hash[:16]}… not in scores.")
        return self.test_index[test_hash]

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, out_dir: Union[str, Path]) -> Path:
        """Persist to ``out_dir`` as ``scores.pt`` + ``metadata.json``.

        Args:
            out_dir: Destination directory; created if missing.

        Returns:
            The directory the artifacts were written to.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.scores, out / self._SCORES_FILE)
        meta = {
            "row_train_ids": self.row_train_ids,
            "row_steps": self.row_steps,
            "test_ids": self.test_ids,
            "algorithm": self.algorithm,
            "algorithm_meta": self.algorithm_meta,
            "normalized_grad": self.normalized_grad,
            "layer_name": self.layer_name,
            "num_train": self.num_train,
            "num_test": self.num_test,
            "num_rows": self.num_rows,
        }
        with open(out / self._META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        return out

    @classmethod
    def load(cls, out_dir: Union[str, Path]) -> "AttributionScore":
        """Reload an :class:`AttributionScore` previously written by :meth:`save`.

        Args:
            out_dir: Directory containing ``scores.pt`` and ``metadata.json``.

        Returns:
            The reconstructed :class:`AttributionScore` (index maps rebuilt).
        """
        out = Path(out_dir)
        scores = torch.load(out / cls._SCORES_FILE, weights_only=True)
        with open(out / cls._META_FILE) as f:
            meta = json.load(f)
        algorithm_meta = meta.get("algorithm_meta", {})
        return cls(
            scores=scores,
            row_train_ids=meta["row_train_ids"],
            row_steps=meta["row_steps"],
            test_ids=meta["test_ids"],
            algorithm=meta["algorithm"],
            algorithm_meta=algorithm_meta,
            normalized_grad=meta["normalized_grad"],
            layer_name=meta["layer_name"],
        )
