"""K-FAC / EK-FAC influence attribution.

Unlike TracIn (a raw inner product), these attributors precondition the inner
product by an approximate inverse Fisher estimated *from the training gradients
themselves*.  They are **single-checkpoint** methods: every record in the train
store is one training sample, every record in the test store one test sample,
and the score is the full ``(num_train, num_test)`` matrix

    score[i, j] = sum_layer  vec(dW_te,j)^T F_l^-1 vec(dW_tr,i)

where the per-layer Fisher is approximated with the Kronecker structure
``F_l ~ A_l x G_l`` (``A`` the input-activation covariance, ``G`` the
output-gradient covariance) fit over the whole training set.

* **K-FAC** uses ``F_l^-1 ~ (A_l + lambda)^-1 x (G_l + lambda)^-1``.
* **EK-FAC** rotates into the Kronecker eigenbasis ``U_A, U_G`` and replaces the
  Kronecker eigenvalues with the *empirical* second moments ``Lambda`` of the
  projected gradients (a second pass over the training gradients), giving
  ``F_l^-1 ~ (U_A x U_G) (Lambda + lambda)^-1 (U_A x U_G)^T``.

Both inverses are symmetric, so the **whole preconditioner is applied on the
(small) test side once**: :meth:`KroneckerAttributor.transform_test_rep` turns
a test block into dense preconditioned per-layer representations, and the
score is then the plain layerwise inner product against the raw train
gradients (the inherited :meth:`inner_product`), with each train layer
materialized once per block by the scoring loop's dense cache.

Only linear and convolution layers are K-FAC-eligible; normalisation and
embedding layers (for which K-FAC is undefined) are skipped by default.  Token/
spatial positions are summed (matching a sum-over-tokens loss).

Normalisation layers are not heavily parametrised, so their per-layer Fisher can
be estimated **directly** rather than with the Kronecker factorisation.  Passing
``non_kfac_strategy="direct"`` adds a dense empirical-Fisher preconditioner
``F_l^-1`` for each such layer (built from the token-summed ``(B, d)`` weight
gradients), whose contribution is summed into the K-FAC score.  Layers stored
**materialized** -- e.g. a TRAK-projected capture, a dense ``(B, proj_dim)``
tensor with no ``(a, g)`` factors -- can never enter K-FAC; they are **always**
preconditioned by the direct dense Fisher (with a warning), regardless of
``non_kfac_strategy``, which governs the norm layers only.  Layers whose
parameter count exceeds ``direct_fim_max_params`` are left out to bound the
``O(d^2)`` Fisher; factorized embedding layers (heavily parametrised) stay
ignored.
"""

from __future__ import annotations

import contextlib
import tempfile
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from dattri_llm.attribution.base import BaseInnerProductAttributor
from dattri_llm.gradient import ops
from dattri_llm.gradient.gradient import Factorized, Gradient
from dattri_llm.gradient.storage_manager import GradientStorageManager
from dattri_llm.utils.cache import CACHE_RESIDENCIES

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dattri.task import AttributionTask
    from torch.utils.data import Dataset

    from dattri_llm.attribution.arguments import AttributionArguments
    from dattri_llm.attribution.score import AttributionScore
    from dattri_llm.gradient.hooks import HookManagerConfig
    from dattri_llm.gradient.streaming import DiskGradientSource, GradientSource

NonKfacStrategy = Literal["ignore", "direct"]

# A raw (damping-free) fit: the per-layer K-FAC factors and the per-layer
# direct empirical Fishers, both dataset-size- and damping-independent.
RawFit = tuple[dict, dict[str, torch.Tensor]]
# The damped, scoring-ready form of a RawFit.
Preconditioner = tuple[dict, dict[str, torch.Tensor]]


class KroneckerAttributor(BaseInnerProductAttributor):
    """Shared workflow of the K-FAC family; subclass to add a Kronecker variant.

    A subclass implements three things:

    * :meth:`fit_factors` -- the **damping-free** per-layer factors from a
      re-iterable train source (K-FAC: the covariances ``(A, G)``; EK-FAC:
      the eigenbases and undamped empirical spectrum).
    * :meth:`damp` -- fold a damping into those factors (data-free).
    * :meth:`precondition_test_layer` -- apply a layer's damped inverse to a
      test block's layer, returning the dense preconditioned ``(B, D)`` rep.

    Everything else is shared: the direct dense-Fisher fallback for the
    non-K-FAC layers, the persisted fit (:meth:`fit` / ``fisher_dir``), the
    preconditioned-test store, and -- through the base class -- collection,
    the scoring loop and the score assembly.  The fit for a scoring pass is
    computed in :meth:`prepare_scoring` unless a persisted one was loaded.
    """

    algorithm: ClassVar[str] = "Kronecker"

    def __init__(
        self,
        args: AttributionArguments,
        *,
        task: AttributionTask | None = None,
    ) -> None:
        super().__init__(args, task=task)
        # Per-call options (set by the entry points, read by prepare_scoring).
        self._damping: float = 1e-3
        self._non_kfac_strategy: NonKfacStrategy = "ignore"
        self._direct_fim_max_params: int = 4096
        # A loaded/fit raw fit, and its damped form for the current pass.
        self._raw_fit: RawFit | None = None
        self._preconditioner: Preconditioner | None = None
        # Per-fit bookkeeping: embedding layers the direct Fisher left
        # uncovered, K-FAC-typed layers diverted to the dense Fisher because
        # they were stored materialized, and whether norm layers enter the
        # Fisher too (the ``non_kfac_strategy="direct"`` choice).
        self._fisher_saw_embedding: set[str] = set()
        self._skipped_materialized: set[str] = set()
        self._direct_norm_layers: bool = False

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit_factors(
        self,
        train_source: GradientSource,
        fisher_acc: ops.FisherAccumulator,
    ) -> dict:
        """Estimate the **damping-free** per-layer K-FAC factors from the
        training gradients.

        Iterates ``train_source`` (a re-iterable ``GradientSource`` -- disk or a
        frozen streamer; EK-FAC iterates it twice).  Returns ``{layer: factors}``
        (possibly empty if no K-FAC-eligible layer is present) that is
        dataset-*size*-independent and damping-independent, so it can be
        persisted once and re-damped per attribution.  :meth:`damp` folds the
        damping in afterwards.

        The implementation must feed every block to *fisher_acc* (via
        :meth:`accumulate_fisher`) during its **first** pass, so the
        direct-Fisher estimate for the non-K-FAC layers reuses that sweep.
        """

    @abstractmethod
    def damp(self, raw_factors: dict, damping: float) -> dict:
        """Fold *damping* into the raw factors from :meth:`fit_factors`.

        Cheap and data-free (per-layer small-matrix inverse / eigenvalue
        shift), so a persisted raw fit can be re-damped for any ``damping``
        without re-streaming the training gradients.
        """

    @abstractmethod
    def precondition_test_layer(
        self,
        value: Factorized | torch.Tensor,
        layer_type: str,
        factors: object,
    ) -> torch.Tensor:
        """Dense ``(B_te, D)`` representation of one test layer with the damped
        inverse Fisher of *factors* applied.
        """

    # ------------------------------------------------------------------ #
    # Fitting                                                             #
    # ------------------------------------------------------------------ #

    def checkpoints(self) -> list[int]:
        """K-FAC/EK-FAC are single-checkpoint: only the task's first is used."""
        n_ckpt = self.num_checkpoints()
        if n_ckpt > 1:
            warnings.warn(
                f"{type(self).__name__} is single-checkpoint; only checkpoint 0 is "
                f"used and the other {n_ckpt - 1} provided checkpoint(s) are ignored.",
                stacklevel=2,
            )
        return [0]

    def fit_raw(self, train_source: GradientSource) -> RawFit:
        """Sweep the training gradients into the **damping-free** raw fit.

        One (or, for EK-FAC, two) sweep(s) over ``train_source``.  The
        empirical-Fisher accumulator is filled in the *same* first pass: it
        always receives layers stored materialized (K-FAC is impossible for
        them, so the dense Fisher is their only preconditioner), and
        additionally the norm layers when the direct strategy is requested.

        Returns ``(raw_factors, raw_fisher)`` -- exactly what :meth:`fit`
        persists; :meth:`damp_fit` turns it into the scored preconditioner.
        All fit-time warnings and the no-eligible-layers check fire here.
        """
        if not getattr(train_source, "reusable", False):
            raise ValueError(
                f"{type(self).__name__} needs a re-iterable train source: the "
                "Fisher pre-pass re-reads the train gradients before scoring. "
                "Use on-disk gradients, a frozen GradientStreamer "
                "(enable_update=False), or attribute(gradient_cache_residency=...).",
            )
        self._fisher_saw_embedding = set()
        self._skipped_materialized = set()
        self._direct_norm_layers = self._non_kfac_strategy == "direct"
        max_params = self._direct_fim_max_params
        fisher_acc = ops.FisherAccumulator(max_params)
        raw_factors = self.fit_factors(train_source, fisher_acc)
        if self._skipped_materialized:
            warnings.warn(
                "Layers stored materialized (e.g. a TRAK-projected capture) "
                "cannot enter K-FAC -- there are no factorized (a, g) factors "
                "to build the Kronecker covariances from.  Preconditioning "
                "them with the direct dense empirical Fisher (FIM) instead, "
                f"bounded by direct_fim_max_params={max_params}: "
                f"{sorted(self._skipped_materialized)}.  Collect with "
                "factorize=True (LoGRA) to keep them K-FAC-eligible.",
                stacklevel=2,
            )
        raw_fisher = self._finalize_fisher_raw(fisher_acc, max_params)
        if not raw_factors and not raw_fisher:
            raise ValueError(
                "No eligible layers found in the training gradients: no "
                "K-FAC-eligible (linear/conv) layer"
                + (
                    " and no direct-Fisher (norm) layer within "
                    f"direct_fim_max_params={max_params}"
                    if self._non_kfac_strategy == "direct"
                    else " (pass non_kfac_strategy='direct' to include norm layers)"
                )
                + ". Check the hook config and the collected layers.",
            )
        return raw_factors, raw_fisher

    def damp_fit(self, raw_fit: RawFit, damping: float) -> Preconditioner:
        """Fold *damping* into a raw fit -> the scoring preconditioner.

        Data-free: per-layer small-matrix inverses / eigenvalue shifts, so a
        persisted raw fit is re-damped for any ``damping`` without touching the
        training gradients.  The dense empirical-Fisher blocks are large (up
        to ``direct_fim_max_params`` wide); they are inverted with Cholesky
        rather than the eigendecomposition :meth:`damp` uses for the small
        K-FAC factors.
        """
        raw_factors, raw_fisher = raw_fit
        return (
            self.damp(raw_factors, damping),
            {layer: ops.dense_inverse(F, damping) for layer, F in raw_fisher.items()},
        )

    def prepare_scoring(
        self,
        train_source: GradientSource,
        test_source: GradientSource,  # noqa: ARG002 - the fit reads the train side
    ) -> None:
        """Fit (unless a persisted fit was loaded) and damp the preconditioner."""
        if self._preconditioner is not None:
            return
        if self._raw_fit is None:
            self._raw_fit = self.fit_raw(train_source)
        self._preconditioner = self.damp_fit(self._raw_fit, self._damping)

    def _set_options(
        self,
        damping: float,
        non_kfac_strategy: NonKfacStrategy,
        direct_fim_max_params: int,
    ) -> None:
        """Validate and store the per-call fit/damping options; reset the pass."""
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
        self._damping = damping
        self._non_kfac_strategy = non_kfac_strategy
        self._direct_fim_max_params = direct_fim_max_params
        self._raw_fit = None
        self._preconditioner = None

    # ------------------------------------------------------------------ #
    # Test-side preconditioning (the transform hook)                      #
    # ------------------------------------------------------------------ #

    def transform_test_rep(self, test_rep: Gradient) -> Gradient:
        """Apply the **entire** preconditioner to a test block, once.

        Every K-FAC layer becomes its dense preconditioned ``(B_te, D)`` rep
        (:meth:`precondition_test_layer`), every direct-Fisher layer its
        ``F^-1``-multiplied dense weight gradient, and layers under neither
        are dropped.  Scoring against a raw train block is then the plain
        layerwise inner product -- no per-train-block whitening or rotation.
        """
        if self._preconditioner is None:
            raise RuntimeError(
                "prepare_scoring() must run before transform_test_rep()."
            )
        factors, fisher_inverse = self._preconditioner

        def precondition(
            name: str,
            value: Factorized | torch.Tensor,
            layer_type: str,
        ) -> torch.Tensor | None:
            if name in factors:
                return self.precondition_test_layer(value, layer_type, factors[name])
            if name in fisher_inverse:
                # F_l^-1 is symmetric, so it is applied wholly on this side.
                dense = ops.materialize(value, layer_type).float()
                return dense @ fisher_inverse[name]
            return None

        return test_rep.map_layers(precondition)

    # ------------------------------------------------------------------ #
    # Direct-Fisher fallback for non-K-FAC layers                          #
    # ------------------------------------------------------------------ #

    def kfac_layers(self, grad: Gradient) -> list[str]:
        """Layer names eligible for K-FAC: linear/conv **stored factorized**.

        A layer of eligible type stored materialized -- e.g. a TRAK-projected
        capture, which keeps its layer type but holds a dense ``(B, proj_dim)``
        tensor -- has no ``(a, g)`` factors to build the covariances from.
        Such layers are recorded (warned about once per fit) and left to the
        direct-Fisher fallback.
        """
        names = []
        for name, value in grad.data.items():
            if not ops.is_kfac_eligible(grad.layer_types[name]):
                continue
            if isinstance(value, Factorized):
                names.append(name)
            else:
                self._skipped_materialized.add(name)
        return names

    def fisher_layers(self, grad: Gradient) -> list[str]:
        """Layer names entering the dense empirical Fisher (FIM).

        Layers stored **materialized** enter unconditionally (the dense Fisher
        is their only preconditioner); norm layers only under
        ``non_kfac_strategy="direct"``.  Batch-level ``param_grad`` tensors
        carry no per-sample axis and are excluded.
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

    def accumulate_fisher(
        self,
        fisher_acc: ops.FisherAccumulator,
        grad: Gradient,
    ) -> None:
        """Fold one streamed training block into the per-layer Fisher estimate.

        Called from :meth:`fit_factors`' first pass.  Also records factorized
        embedding layers so the fit can warn that they were left ignored
        (heavily parametrised -- not covered by the direct fallback).
        """
        fisher_acc.update(grad, self.fisher_layers(grad))
        if self._direct_norm_layers:
            self._fisher_saw_embedding.update(
                name
                for name, value in grad.data.items()
                if ops.is_embedding(grad.layer_types[name])
                and isinstance(value, Factorized)
            )

    def _finalize_fisher_raw(
        self,
        fisher_acc: ops.FisherAccumulator,
        max_params: int,
    ) -> dict[str, torch.Tensor]:
        """The accumulated **undamped** Fishers ``{layer: F}``, warning about the
        layers dropped by the cap and the ignored embeddings.
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
        return dict(fisher_acc.result())

    # ------------------------------------------------------------------ #
    # Persisted (dataset-size-independent) fit                             #
    # ------------------------------------------------------------------ #

    _FISHER_FILE = "fisher_factors.pt"

    def _fisher_meta(self) -> dict:
        """Identity of a persisted fit -- checked on load for compatibility."""
        meta = {
            "algorithm": self.algorithm,
            "non_kfac_strategy": self._non_kfac_strategy,
            "direct_fim_max_params": self._direct_fim_max_params,
        }
        mode = getattr(self, "mode", None)
        if mode is not None:
            meta["mode"] = mode
        return meta

    @staticmethod
    def _move_raw(raw: dict, device: torch.device) -> dict:
        """Move a raw-factor dict (``{layer: tensor}`` / ``{layer: tuple}``)."""
        return {
            layer: (
                tuple(t.to(device) for t in value)
                if isinstance(value, tuple)
                else value.to(device)
            )
            for layer, value in raw.items()
        }

    def _default_fisher_dir(self) -> str:
        return str(Path(self.args.output_dir) / f"{self.algorithm.lower()}_fisher")

    def save_fisher(
        self,
        covariances: dict,
        fisher_dir: str | None = None,
        *,
        fisher: dict[str, torch.Tensor] | None = None,
    ) -> str:
        """Persist raw (damping-free) factors as a fit :meth:`attribute_from_cache`
        can load through ``fisher_dir``.

        *covariances* is the ``{layer: (A, G)}`` produced during collection by a
        :class:`~dattri_llm.gradient.callbacks.KroneckerCovarianceCallback`
        (manual workflow) or a :meth:`cache` ``on_train_block`` accumulator
        (on-the-fly) -- or, for EK-FAC, that attributor's own raw factors.
        Writing them in the format :meth:`fit` uses lets later attribution
        re-damp and score **without a Fisher pre-pass**; when the gradients
        were captured ``logra_materialized``, the ``(A, G)`` are the compact
        projected covariances that precondition that compact store directly.

        Args:
            covariances: Raw ``{layer: factors}`` for the K-FAC layers.
            fisher_dir: Where to write them; defaults to
                ``<args.output_dir>/<algorithm>_fisher``.
            fisher: Raw ``{layer: F}`` direct empirical Fishers (none by
                default).

        Returns:
            ``fisher_dir``.
        """
        if fisher_dir is None:
            fisher_dir = self._default_fisher_dir()
        path = Path(fisher_dir)
        path.mkdir(parents=True, exist_ok=True)
        cpu = torch.device("cpu")
        # Persist on CPU so the factor file is portable across devices.
        torch.save(
            {
                "meta": self._fisher_meta(),
                "raw_ctx": self._move_raw(covariances, cpu),
                "raw_fim": self._move_raw(fisher or {}, cpu),
            },
            path / self._FISHER_FILE,
        )
        return fisher_dir

    def load_fisher(self, fisher_dir: str) -> RawFit:
        """Load and validate a persisted fit, placed on ``args.device``."""
        path = Path(fisher_dir) / self._FISHER_FILE
        if not path.exists():
            raise ValueError(
                f"No fitted Fisher factors at {path}; build them with "
                f"{type(self).__name__}.fit(train_gradients_dir, "
                f"{fisher_dir!r}) first.",
            )
        blob = torch.load(path, map_location="cpu", weights_only=True)
        meta = blob["meta"]
        if meta.get("algorithm") != self.algorithm:
            raise ValueError(
                f"fisher_dir {fisher_dir!r} holds {meta.get('algorithm')!r} "
                f"factors, but this is a {self.algorithm} attributor.",
            )
        mode = getattr(self, "mode", None)
        if mode is not None and meta.get("mode") != mode:
            raise ValueError(
                f"fisher_dir {fisher_dir!r} was fit with mode="
                f"{meta.get('mode')!r}, but this attributor uses mode={mode!r}.",
            )
        device = self.args.device
        return self._move_raw(blob["raw_ctx"], device), self._move_raw(
            blob["raw_fim"],
            device,
        )

    def fit(
        self,
        train_gradients_dir: str,
        fisher_dir: str | None = None,
        *,
        selected_training_steps: Iterable[int] | None = None,
        non_kfac_strategy: NonKfacStrategy = "ignore",
        direct_fim_max_params: int = 4096,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """Fit and persist the **damping-free** Fisher factors, once.

        The fitted factors (K-FAC covariances ``A``/``G``, EK-FAC eigenbases
        ``U_A``/``U_G`` + undamped empirical spectrum, and any direct dense
        Fisher) are dataset-*size*-independent, so persisting them lets later
        re-attribution -- **with new queries, or a different ``damping``** --
        skip the Fisher pre-pass entirely.  Pass the returned directory to
        ``attribute_from_cache(..., fisher_dir=...)``.

        Args:
            train_gradients_dir: Directory written by :class:`GradientStorageManager`
                for the train pass (the gradients the Fisher is fit from).
            fisher_dir: Where to write the factors; defaults to
                ``<args.output_dir>/<algorithm>_fisher``.
            selected_training_steps: Restrict the fit to these train steps.
            non_kfac_strategy: Fixes which non-K-FAC (norm) layers enter the
                direct dense Fisher.  Recorded in the factor file.
            direct_fim_max_params: Parameter-count cap for the dense Fisher.
            layer_name: Restrict the fit to this subset of the stored layers.
            verbose: Show progress bars on the logging process.

        Returns:
            ``fisher_dir``.
        """
        self._set_options(self._damping, non_kfac_strategy, direct_fim_max_params)
        train = self.load_train_rep(
            train_gradients_dir,
            steps=selected_training_steps,
            layer_name=layer_name,
            verbose=verbose,
            desc=f"{self.algorithm}: fitting Fisher",
        )
        raw_factors, raw_fisher = self.fit_raw(train)
        return self.save_fisher(raw_factors, fisher_dir, fisher=raw_fisher)

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
        loop_over_test: bool = False,
        gradient_cache_residency: str | None = None,
        damping: float = 1e-3,
        non_kfac_strategy: NonKfacStrategy = "ignore",
        direct_fim_max_params: int = 4096,
    ) -> AttributionScore:
        """Score by collecting gradients **live** at the task's first checkpoint.

        Both sides are frozen probes.  ``gradient_cache_residency=None``
        (default) streams the gradients straight into the fit + scoring
        passes, re-running the model for each pass (K-FAC twice, EK-FAC three
        times) -- cheapest when the captures are too large to hold.
        ``"memory"``/``"tiered"``/``"disk"`` collects each side **once** into a
        store of that residency and reads it back on every pass; under
        ``loop_over_test=True`` the preconditioned test representations are
        cached in the same residency too.

        Args:
            train_dataset: Training dataset to stream.
            test_dataset: Test dataset to stream.
            hook_config: Capture configuration for the internal streamers.
            verbose: Accepted for API parity.
            loop_over_test: Re-stream the test blocks per train block.
            gradient_cache_residency: See above.
            damping: Tikhonov term added to each covariance factor (K-FAC) or
                to the corrected eigenvalues (EK-FAC) before inversion.
            non_kfac_strategy: ``"ignore"`` (default) skips norm layers;
                ``"direct"`` preconditions them with a dense empirical Fisher.
            direct_fim_max_params: Parameter-count cap for that dense Fisher.
        """
        self._set_options(damping, non_kfac_strategy, direct_fim_max_params)
        extra: dict = {}
        if gradient_cache_residency is not None and loop_over_test:
            extra["preconditioned_test_cache_residency"] = gradient_cache_residency
        return super().attribute(
            train_dataset,
            test_dataset,
            hook_config=hook_config,
            verbose=verbose,
            loop_over_test=loop_over_test,
            enable_update=False,
            gradient_cache_residency=gradient_cache_residency,
            damping=damping,
            non_kfac_strategy=non_kfac_strategy,
            direct_fim_max_params=direct_fim_max_params,
            **extra,
        )

    def attribute_from_cache(
        self,
        train_source: str | Path | GradientStorageManager | DiskGradientSource,
        test_source: str | Path | GradientStorageManager | DiskGradientSource,
        *,
        selected_training_steps: Iterable[int] | None = None,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
        loop_over_test: bool = False,
        algorithm_meta: dict | None = None,
        damping: float = 1e-3,
        preconditioned_test_dir: str | None = None,
        preconditioned_test_cache_residency: str | None = None,
        fisher_dir: str | None = None,
        non_kfac_strategy: NonKfacStrategy = "ignore",
        direct_fim_max_params: int = 4096,
    ) -> AttributionScore:
        """Score collected gradients (the *store-then-attribute* path).

        The Fisher is estimated from the (selected) train gradients unless
        *fisher_dir* supplies a persisted fit (see :meth:`fit`), which is
        loaded and re-damped instead -- the train pre-pass is skipped.

        Args:
            train_source: Train gradients -- directory, open store, or source.
            test_source: Test gradients, likewise.
            selected_training_steps: Restrict the train steps (Fisher fit +
                output rows) to these.
            layer_name: Restrict scoring (and the fit) to these stored layers.
            verbose: Show progress bars on the logging process.
            loop_over_test: Re-stream + rebuild the test reps per train block
                (low memory) instead of caching them once (default).
            algorithm_meta: Extra entries for the score's metadata.
            damping: As in :meth:`attribute`.
            preconditioned_test_dir: With ``loop_over_test=True``, persist the
                preconditioned test representations to this **on-disk**
                directory and re-stream them from disk on every sweep instead
                of recomputing them from the raw test gradients each time.
                Durable -- see :meth:`cache_preconditioned_test` to build it
                ahead of time.  Takes precedence over the residency below.
            preconditioned_test_cache_residency: With ``loop_over_test=True``
                and no ``preconditioned_test_dir``, cache the preconditioned
                test representations in an **ephemeral** store of this
                residency (``"memory"``/``"tiered"``/``"disk"`` temp),
                released on return.  ``None`` (default) recomputes per block.
            fisher_dir: Directory of factors persisted by :meth:`fit` /
                :meth:`save_fisher`.  When given, ``non_kfac_strategy`` and
                ``direct_fim_max_params`` come from the recorded fit.
            non_kfac_strategy: As in :meth:`attribute`.
            direct_fim_max_params: As in :meth:`attribute`.
        """
        self._set_options(damping, non_kfac_strategy, direct_fim_max_params)
        cache_precond = (
            preconditioned_test_dir is not None
            or preconditioned_test_cache_residency is not None
        )
        if cache_precond and not loop_over_test:
            raise ValueError(
                "preconditioned_test_dir / preconditioned_test_cache_residency "
                "only apply to loop_over_test=True (with loop_over_test=False the "
                "preconditioned representations are simply held in memory).",
            )
        if (
            preconditioned_test_cache_residency is not None
            and preconditioned_test_cache_residency not in CACHE_RESIDENCIES
        ):
            raise ValueError(
                "preconditioned_test_cache_residency must be one of "
                f"{list(CACHE_RESIDENCIES)} or None, got "
                f"{preconditioned_test_cache_residency!r}.",
            )
        if fisher_dir is not None:
            self._raw_fit = self.load_fisher(fisher_dir)
        train_store = self.resolve_store(train_source)
        test_store = self.resolve_store(test_source)
        meta = {
            "damping": damping,
            "non_kfac_strategy": non_kfac_strategy,
            "fisher_dir": fisher_dir,
            **(algorithm_meta or {}),
        }
        if not cache_precond:
            result = super().attribute_from_cache(
                train_store,
                test_store,
                selected_training_steps=selected_training_steps,
                layer_name=layer_name,
                verbose=verbose,
                loop_over_test=loop_over_test,
                algorithm_meta=meta,
            )
            return self._stamp_direct_layers(result)

        # Precondition once and re-stream the (small) store per train block.  An
        # explicit dir gives a durable disk store; otherwise an ephemeral
        # residency store (context-managed, so a tiered spill is cleaned up).
        train = self.load_train_rep(
            train_store,
            steps=selected_training_steps,
            layer_name=layer_name,
            verbose=verbose,
        )
        test = self.load_test_rep(test_store, layer_name=layer_name, verbose=verbose)
        self.prepare_scoring(train, test)
        with contextlib.ExitStack() as stack:
            if preconditioned_test_dir is not None:
                store = GradientStorageManager(preconditioned_test_dir)
            else:
                store = stack.enter_context(
                    GradientStorageManager(
                        tempfile.mkdtemp(prefix=f"{self.algorithm.lower()}_precond_"),
                        residency=preconditioned_test_cache_residency,
                    ),
                )
            self.cache_representations(
                test, store, sample_id_key=test_store.sample_id_key
            )
            precond = self.load_test_rep(
                store,
                desc=f"{self.algorithm}: preconditioned test",
            )
            scores, row_train_ids, row_steps, test_ids = self.score_sources(
                train,
                precond,
                loop_over_test=True,
                transform_test=lambda block: block,  # already preconditioned
            )
        result = self.build_score(
            scores,
            row_train_ids,
            row_steps,
            test_ids,
            algorithm_meta={
                "selected_training_steps": train.steps,
                **self.stores_meta(train_store, test_store),
                **meta,
            },
            layer_name=train.layer_name,
        )
        return self._stamp_direct_layers(result)

    def _stamp_direct_layers(self, result: AttributionScore) -> AttributionScore:
        """Record which layers went through the direct Fisher in the score."""
        if self._preconditioner is not None:
            result.algorithm_meta["direct_fim_layers"] = sorted(self._preconditioner[1])
        return result

    def cache_preconditioned_test(
        self,
        train_gradients_dir: str,
        test_gradients_dir: str,
        preconditioned_test_dir: str | None = None,
        *,
        damping: float = 1e-3,
        selected_training_steps: Iterable[int] | None = None,
        non_kfac_strategy: NonKfacStrategy = "ignore",
        direct_fim_max_params: int = 4096,
        layer_name: str | list[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """Fit the preconditioner and persist **preconditioned** test reps.

        A one-time sweep that turns raw test gradients into a store already
        carrying the full K-FAC/EK-FAC (and direct-Fisher) preconditioner on
        the test side.  Scoring the store against train gradients then reduces
        to a plain ``TracInAttributor`` inner product -- no preconditioner is
        recomputed::

            pre_dir = attr.cache_preconditioned_test(train_dir, test_dir)
            scores = TracInAttributor(args).attribute_from_cache(train_dir, pre_dir)

        The store can also be passed to ``attribute_from_cache(...,
        loop_over_test=True, preconditioned_test_dir=...)``.

        Args:
            train_gradients_dir: Train store (fits the preconditioner).
            test_gradients_dir: Test store (the raw gradients preconditioned).
            preconditioned_test_dir: Where to store the result; defaults to
                ``<args.output_dir>/<algorithm>_preconditioned_test``.
            damping: As in :meth:`attribute`.
            selected_training_steps: Restricts the fit, not what is stored.
            non_kfac_strategy: As in :meth:`attribute`.
            direct_fim_max_params: As in :meth:`attribute`.
            layer_name: As in :meth:`attribute_from_cache`.
            verbose: Show progress bars on the logging process.

        Returns:
            ``preconditioned_test_dir``.
        """
        if preconditioned_test_dir is None:
            subdir = f"{self.algorithm.lower()}_preconditioned_test"
            preconditioned_test_dir = str(Path(self.args.output_dir) / subdir)
        self._set_options(damping, non_kfac_strategy, direct_fim_max_params)
        train = self.load_train_rep(
            train_gradients_dir,
            steps=selected_training_steps,
            layer_name=layer_name,
            verbose=verbose,
            desc=f"{self.algorithm}: train (fitting)",
        )
        test_store = GradientStorageManager(test_gradients_dir)
        test = self.load_test_rep(
            test_store,
            layer_name=layer_name,
            verbose=verbose,
            desc=f"{self.algorithm}: test (raw)",
        )
        self.prepare_scoring(train, test)
        self.cache_representations(
            test,
            GradientStorageManager(preconditioned_test_dir),
            sample_id_key=test_store.sample_id_key,
        )
        return preconditioned_test_dir


class KFACAttributor(KroneckerAttributor):
    """K-FAC influence attributor.

    ``F_l^-1 ~ (A_l + lambda)^-1 x (G_l + lambda)^-1`` per linear/conv layer,
    with ``lambda`` the ``damping`` term (a per-attribution argument).  Because
    the per-sample gradient factorises as ``sum_t g_t a_t^T``, the inverse is
    applied two-sided to the (materialized) test gradient in one step.

    Args:
        args: :class:`AttributionArguments`.
        task: The attribution task; required by the live methods only.
    """

    algorithm: ClassVar[str] = "KFAC"

    def fit_factors(
        self,
        train_source: GradientSource,
        fisher_acc: ops.FisherAccumulator,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """One sweep: the raw ``{layer: (A, G)}`` covariances (undamped)."""
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(self.args.device)
            kron.update(train_g, self.kfac_layers(train_g))
            # Reuse this single sweep to fit the direct Fisher.
            self.accumulate_fisher(fisher_acc, train_g)
        return kron.result()  # {layer: (A, G)} raw covariances (undamped)

    def damp(  # noqa: PLR6301 - subclass hook
        self,
        raw_factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
        damping: float,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """``{layer: (A_inv, G_inv)}`` -- the damped inverse covariances."""
        # The inverses are the only place damping enters K-FAC scoring, so
        # re-damping is two small eighs per layer -- no training sweep.
        return {
            layer: (ops.sym_inverse(A, damping), ops.sym_inverse(G, damping))
            for layer, (A, G) in raw_factors.items()
        }

    def precondition_test_layer(  # noqa: PLR6301 - subclass hook
        self,
        value: Factorized | torch.Tensor,
        layer_type: str,
        factors: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``vec(G_inv dW A_inv)`` per test sample, as a dense ``(B_te, D)``."""
        # Materialize the (fixed) test layer ONCE -- token-sum the factors to
        # the full weight gradient (or take the compact materialized block as
        # is) -- then apply the whole inverse two-sided in that space.
        A_inv, G_inv = factors
        return ops.kfac_precondition_materialized(
            ops.materialize(value, layer_type),
            A_inv,
            G_inv,
        )


class EKFACAttributor(KroneckerAttributor):
    """EK-FAC influence attributor.

    Rotates each layer's gradients into the Kronecker eigenbasis ``(U_A, U_G)``
    and replaces the Kronecker eigenvalues with the empirical second moments
    ``Lambda`` of the projected gradients (a second pass over the training
    gradients), giving ``F_l^-1 ~ (U_A x U_G)(Lambda + lambda)^-1(U_A x U_G)^T``.

    The per-sample gradient is projected as ``M = U_G^T dW U_A`` -- the faithful
    expansion of ``(U_A x U_G)^T vec(dW)``.  This is the unique projection that
    **reduces to K-FAC** when ``Lambda`` equals the Kronecker eigenvalues, and it
    is invariant to the (arbitrary) sign of each eigenvector.

    ``mode`` selects the implementation and is kept for backward compatibility /
    cross-checking: ``"exact"`` (default) is the faithful projection above;
    ``"approx"`` mirrors the ``dattri`` library's code path, which after its
    projection fix produces identical scores.

    Args:
        args: :class:`AttributionArguments`.
        task: The attribution task; required by the live methods only.
        mode: ``"exact"`` (default) or ``"approx"``; currently equivalent.
    """

    algorithm: ClassVar[str] = "EKFAC"
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

    def fit_factors(
        self,
        train_source: GradientSource,
        fisher_acc: ops.FisherAccumulator,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Two sweeps: ``{layer: (U_A, U_G, lambda_raw)}`` -- the Kronecker
        eigenbases and the undamped empirical spectrum in that basis.
        """
        device = self.args.device
        # Pass 1 -- Kronecker covariance factors and their eigenbases (and the
        # direct Fisher from the same sweep).
        kron = ops.KroneckerAccumulator()
        for _step, train_block, _ in train_source:
            train_g = train_block.to(device)
            kron.update(train_g, self.kfac_layers(train_g))
            self.accumulate_fisher(fisher_acc, train_g)
        eig: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer, (A, G) in kron.result().items():
            _, U_A, _, U_G = ops.kfac_eigh(A, G)
            eig[layer] = (U_A, U_G)

        # Pass 2 -- empirical second moments of the projected gradients (Lambda).
        # Skipped entirely when no K-FAC layer is present.
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

        # The *undamped* empirical spectrum; damping is a per-layer shift
        # applied later in damp(), so the raw fit can be re-damped freely.
        return {
            layer: (U_A, U_G, lam_sum[layer] / counts[layer])
            for layer, (U_A, U_G) in eig.items()
        }

    def damp(  # noqa: PLR6301 - subclass hook
        self,
        raw_factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        damping: float,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``{layer: (U_A, U_G, lambda_raw + damping)}`` -- a spectrum shift."""
        # F_l^-1 ~ (U_A x U_G)(lambda_raw + damping)^-1(U_A x U_G)^T
        return {
            layer: (U_A, U_G, lam_raw + damping)
            for layer, (U_A, U_G, lam_raw) in raw_factors.items()
        }

    def precondition_test_layer(  # noqa: PLR6301 - subclass hook
        self,
        value: Factorized | torch.Tensor,
        layer_type: str,
        factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """``U_G ((U_G^T dW U_A) / lam) U_A^T`` per test sample, dense ``(B_te, D)``."""
        # Apply the *entire* damped EK-FAC inverse once: rotate into the
        # eigenbasis, divide by the corrected spectrum, and rotate back out
        # (``R = U_G (M/lam) U_A^T``), so raw train gradients dot against R.
        U_A, U_G, lam = factors
        M = ops.ekfac_materialize(value, layer_type, U_A, U_G)  # (B_te, D)
        return ops.ekfac_precondition(M, U_A, U_G, lam, layer_type)
