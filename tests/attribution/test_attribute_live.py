"""Live (on-the-fly) ``attribute()`` workflow tests.

The live workflow runs two ``GradientStreamer``s over one model with a
**shared** ``HookManager`` (the test streamer rides the train streamer's
hooks).  ``loop_over_test=True`` re-streams the test blocks between train
blocks, interleaving the two streamers' passes over the shared manager --
the step bookkeeping must survive that interleaving and produce scores
identical to the cached (``loop_over_test=False``) path.
"""

from __future__ import annotations

import pytest
import torch
from dattri.task import AttributionTask
from torch import nn
from torch.utils.data import Dataset

from dattri_llm.attribution.algorithm.dvemb import DVEmbAttributor
from dattri_llm.attribution.algorithm.kronecker import EKFACAttributor, KFACAttributor
from dattri_llm.attribution.algorithm.tracin import TracInAttributor
from dattri_llm.attribution.arguments import AttributionArguments

IN_DIM, HID_DIM, OUT_DIM = 4, 8, 3
N_TRAIN, N_TEST = 6, 4
SEED = 0


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(IN_DIM, HID_DIM, bias=False))
        self.mlp.add_module("act", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(HID_DIM, OUT_DIM, bias=False))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x)


class DictDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "y": self.y[i]}


def _make_task_and_data():
    torch.manual_seed(SEED)
    model = MLP().eval()
    g = torch.Generator().manual_seed(SEED)
    train_ds = DictDataset(
        torch.randn(N_TRAIN, IN_DIM, generator=g),
        torch.randn(N_TRAIN, OUT_DIM, generator=g),
    )
    test_ds = DictDataset(
        torch.randn(N_TEST, IN_DIM, generator=g),
        torch.randn(N_TEST, OUT_DIM, generator=g),
    )

    def loss_func(params, data):
        yhat = torch.func.functional_call(model, params, (data["x"],))
        return ((yhat - data["y"]) ** 2).sum()

    checkpoint = {k: v.detach().clone() for k, v in model.state_dict().items()}
    task = AttributionTask(loss_func=loss_func, model=model, checkpoints=[checkpoint])
    return task, train_ds, test_ds


def _args(out_dir) -> AttributionArguments:
    # Batch sizes chosen so both sides span multiple blocks: 3 train blocks
    # and 2 test blocks.  loop_over_test=True re-streams the 2 test blocks
    # after every one of the 3 train blocks -- the interleaving under test.
    return AttributionArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        use_cpu=True,
        dataloader_pin_memory=False,
    )


class TestLiveLoopOverTest:
    @pytest.mark.parametrize("normalized_grad", [False, True])
    def test_looped_matches_cached(self, tmp_path, normalized_grad):
        """loop_over_test=True must score identically to the cached path.

        Regression: with the shared HookManager, each test-source re-stream
        reset and advanced the shared step counter mid-train-pass, tripping
        the streamer's step-desync guard on the second train block.
        """
        task, train_ds, test_ds = _make_task_and_data()

        cached = TracInAttributor(_args(tmp_path / "cached"), task=task).attribute(
            train_ds,
            test_ds,
            loop_over_test=False,
            normalized_grad=normalized_grad,
        )
        looped = TracInAttributor(_args(tmp_path / "looped"), task=task).attribute(
            train_ds,
            test_ds,
            loop_over_test=True,
            normalized_grad=normalized_grad,
        )

        ids_c, m_c = cached.agnostic_matrix()
        ids_l, m_l = looped.agnostic_matrix()
        assert ids_c == ids_l
        assert cached.test_ids == looped.test_ids
        assert torch.allclose(m_c, m_l, atol=1e-5), (
            f"max diff {(m_c - m_l).abs().max():.2e}"
        )


class TestAsyncDiskWrite:
    def test_async_cache_scores_match_sync(self, tmp_path):
        """``async_disk_write=True`` is a performance-only knob: the cached
        store and the scores computed from it must match a synchronous run.
        """
        task, train_ds, test_ds = _make_task_and_data()

        def run(out_dir, async_write):
            args = _args(out_dir)
            args.async_disk_write = async_write
            attr = TracInAttributor(args, task=task)
            ((train_dir, test_dir),) = attr.cache(train_ds, test_ds)
            return attr.attribute_from_cache(
                train_gradients_dir=train_dir,
                test_gradients_dir=test_dir,
            )

        sync_res = run(tmp_path / "sync", async_write=False)
        async_res = run(tmp_path / "async", async_write=True)

        ids_s, m_s = sync_res.agnostic_matrix()
        ids_a, m_a = async_res.agnostic_matrix()
        assert ids_s == ids_a
        assert sync_res.test_ids == async_res.test_ids
        assert torch.equal(m_s, m_a), f"max diff {(m_s - m_a).abs().max():.2e}"


class TestCachedGradientsMatchLive:
    """residency={memory,tiered} (collect the raw representations once into a
    store, then replay the Fisher sweeps) must score identically to the default
    residency=None re-streaming path.
    """

    @pytest.mark.parametrize("attr_cls", [KFACAttributor, EKFACAttributor])
    @pytest.mark.parametrize("residency", ["memory", "tiered", "disk"])
    @pytest.mark.parametrize("loop_over_test", [False, True])
    def test_cached_matches_restreamed(
        self,
        tmp_path,
        attr_cls,
        residency,
        loop_over_test,
    ):
        task, train_ds, test_ds = _make_task_and_data()

        # loop_over_test=True re-streams the test set per train block; the cached
        # run then caches the preconditioned test reps in the same residency, so
        # this also covers the residency-managed preconditioned store.
        live = attr_cls(_args(tmp_path / "live"), task=task).attribute(
            train_ds,
            test_ds,
            damping=1e-3,
            loop_over_test=loop_over_test,
        )
        cached = attr_cls(_args(tmp_path / "cached"), task=task).attribute(
            train_ds,
            test_ds,
            damping=1e-3,
            residency=residency,
            loop_over_test=loop_over_test,
        )

        ids_live, m_live = live.agnostic_matrix()
        ids_cached, m_cached = cached.agnostic_matrix()
        assert ids_live == ids_cached
        assert live.test_ids == cached.test_ids
        # loop_over_test=True caches *materialized* preconditioned test reps,
        # so KFAC's whiten-then-materialize adds float32 round-off vs the live
        # factor-domain contraction (EKFAC is already materialized -> exact).
        # loop=False stays in the factor domain and is bit-close.
        atol, rtol = (1e-4, 1e-3) if loop_over_test else (1e-5, 1e-5)
        assert torch.allclose(m_live, m_cached, atol=atol, rtol=rtol), (
            f"max diff {(m_live - m_cached).abs().max():.2e}"
        )


class TestDVEmbResidencyMatchesDisk:
    """DVEmb attribute() must score identically regardless of where the raw
    representations are stored (disk vs memory vs tiered) -- residency only
    relocates the trajectory store, never the sweep math.
    """

    @pytest.mark.parametrize("residency", ["memory", "tiered"])
    def test_residency_matches_disk(self, tmp_path, residency):
        # DVEmb collects an *updating* trajectory (enable_update=True) that
        # mutates the model in place, and its dataloader shuffle consumes global
        # RNG.  So each run needs a fresh task (model reset to init) and the same
        # seed, otherwise the two runs descend different trajectories.  With the
        # trajectory pinned, residency only relocates the store: scores are
        # bit-identical.
        task, train_ds, test_ds = _make_task_and_data()
        torch.manual_seed(SEED)
        disk = DVEmbAttributor(_args(tmp_path / "disk"), task=task).attribute(
            train_ds,
            test_ds,
            residency="disk",
            learning_rate=0.01,
        )
        task, train_ds, test_ds = _make_task_and_data()
        torch.manual_seed(SEED)
        ram = DVEmbAttributor(_args(tmp_path / "ram"), task=task).attribute(
            train_ds,
            test_ds,
            residency=residency,
            learning_rate=0.01,
        )
        ids_d, m_d = disk.agnostic_matrix()
        ids_r, m_r = ram.agnostic_matrix()
        assert ids_d == ids_r
        assert disk.test_ids == ram.test_ids
        assert torch.equal(m_d, m_r), f"max diff {(m_d - m_r).abs().max():.2e}"


class TestCollectToDiskOnBlock:
    """collect_to_disk's on_block hook is the OTF covariance path: accumulating
    K-FAC covariances off the streamed blocks (no callback, no re-pass) must
    reproduce KFACAttributor.fit's covariances over the resulting store.
    """

    def _streamer(self, attr, train_ds):
        from dattri_llm.attribution.utils import task_loss_fn
        from dattri_llm.gradient.streaming import GradientStreamer

        attr.task._load_checkpoints(0)
        return GradientStreamer(
            attr.task.get_model(),
            train_ds,
            attr.args,
            batch_size=attr.args.per_device_train_batch_size,
            loss_fn=task_loss_fn(attr.task.original_loss_func),
        )

    def test_on_block_covariance_matches_fit(self, tmp_path):
        from dattri_llm.attribution.utils import collect_to_disk
        from dattri_llm.gradient.ops import KroneckerAccumulator
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import DiskGradientSource

        task, train_ds, _ = _make_task_and_data()
        attr = KFACAttributor(_args(tmp_path / "o"), task=task)

        fm = GradientStorageManager(str(tmp_path / "store"))
        kron = KroneckerAccumulator()
        n_blocks = 0

        def on_block(_step, grad, _hashes):
            nonlocal n_blocks
            n_blocks += 1
            kron.update(grad, attr._kfac_layers(grad))

        collect_to_disk(self._streamer(attr, train_ds), fm, on_block=on_block)
        otf = kron.result()
        assert n_blocks > 0

        # Re-pass fit over the store the same collection just wrote.
        raw_ctx, _ = attr._fit_raw_context(
            DiskGradientSource(fm, attr.args),
            attr.args.device,
            "ignore",
            4096,
        )
        assert set(otf) == set(raw_ctx)
        for layer in raw_ctx:
            a_o, g_o = otf[layer]
            a_f, g_f = raw_ctx[layer]
            assert torch.allclose(a_o, a_f, atol=1e-5), f"A {layer}"
            assert torch.allclose(g_o, g_f, atol=1e-5), f"G {layer}"

    def test_on_block_none_is_a_noop(self, tmp_path):
        from dattri_llm.attribution.utils import collect_to_disk
        from dattri_llm.gradient.storage_manager import GradientStorageManager

        task, train_ds, _ = _make_task_and_data()
        attr = KFACAttributor(_args(tmp_path / "o"), task=task)
        fm = GradientStorageManager(str(tmp_path / "store"))
        collect_to_disk(self._streamer(attr, train_ds), fm)  # no on_block
        assert fm.index  # still collected the store


class TestCompactKFAC:
    """K-FAC over a logra_materialized (compact) store, preconditioned by the
    projected (A, G) collected at capture, must match K-FAC over a
    logra_factorized store of the same projected gradients -- the logix-style
    compact path.
    """

    PROJ = 4
    DAMP = 1e-3

    def _proj(self, style):
        from dattri_llm.gradient.hooks import REGISTER_ALL, HookManagerConfig

        return HookManagerConfig(
            linear_io=REGISTER_ALL,
            projection={
                "__default__": {
                    "style": style,
                    "proj_dim": self.PROJ,
                    "proj_max_batch_size": 32,
                    "proj_type": "rademacher",
                    "proj_seed": 0,
                },
            },
        )

    def _collect(self, attr, ds, out, style, cov=None):
        from dattri_llm.attribution.utils import collect_to_disk, task_loss_fn
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import GradientStreamer

        attr.task._load_checkpoints(0)
        streamer = GradientStreamer(
            attr.task.get_model(),
            ds,
            attr.args,
            batch_size=attr.args.per_device_train_batch_size,
            loss_fn=task_loss_fn(attr.task.original_loss_func),
            config=self._proj(style),
        )
        if cov is not None:
            streamer.hook_manager.add_callback(cov)
        fm = GradientStorageManager(str(out))
        collect_to_disk(streamer, fm)
        return str(out)

    def test_compact_matches_factorized(self, tmp_path):
        from dattri_llm.gradient.callbacks import KroneckerCovarianceCallback

        # -- factorized reference: fit covariances in a re-pass --
        task, tr, te = _make_task_and_data()
        attr_f = KFACAttributor(_args(tmp_path / "f"), task=task)
        train_f = self._collect(attr_f, tr, tmp_path / "tr_f", "logra_factorized")
        test_f = self._collect(attr_f, te, tmp_path / "te_f", "logra_factorized")
        ids_f, s_fac = attr_f.attribute_from_cache(
            train_f,
            test_f,
            damping=self.DAMP,
        ).agnostic_matrix()

        # -- compact: covariances collected at capture, scored two-sided --
        task, tr, te = _make_task_and_data()
        attr_m = KFACAttributor(_args(tmp_path / "m"), task=task)
        cov = KroneckerCovarianceCallback()
        train_m = self._collect(
            attr_m,
            tr,
            tmp_path / "tr_m",
            "logra_materialized",
            cov=cov,
        )
        test_m = self._collect(attr_m, te, tmp_path / "te_m", "logra_materialized")
        fisher = attr_m.save_fisher(cov.result(), str(tmp_path / "fisher"))
        ids_m, s_mat = attr_m.attribute_from_cache(
            train_m,
            test_m,
            damping=self.DAMP,
            fisher_dir=fisher,
        ).agnostic_matrix()

        assert ids_f == ids_m
        assert torch.allclose(s_fac, s_mat, atol=1e-4, rtol=1e-3), (
            f"max diff {(s_fac - s_mat).abs().max():.2e}"
        )


class TestBatchedScoring:
    """score_sources always re-batches the train side into
    ``per_device_train_batch_size`` groups; the batch size only affects
    speed/memory, so scores must be bit-identical for every value.
    """

    @staticmethod
    def _materialized_store(out, n_blocks, per_block, hash_prefix, d=12):
        from dattri_llm.gradient.gradient import Gradient, GradientRecord
        from dattri_llm.gradient.storage_manager import GradientStorageManager

        fm = GradientStorageManager(str(out))
        for s in range(n_blocks):
            g = Gradient(
                representation={"L0": "materialized", "L1": "materialized"},
                data={"L0": torch.randn(per_block, d), "L1": torch.randn(per_block, d)},
                layer_types={"L0": "nn.Linear", "L1": "nn.Linear"},
            )
            hashes = [f"{hash_prefix}{s}_{i}" for i in range(per_block)]
            fm.save_bulk([GradientRecord(step=s, input_hash=hashes, gradient=g)])
        return str(out)

    def test_batch_size_invariant(self, tmp_path):
        from dattri_llm.attribution.utils import score_sources
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import DiskGradientSource

        torch.manual_seed(0)
        train_dir = self._materialized_store(tmp_path / "tr", 4, 3, "t")  # 12 docs
        test_dir = self._materialized_store(tmp_path / "te", 2, 2, "q")

        def prep(test_g):
            return {name: test_g.data[name] for name in test_g.data}

        def score_block(train_g, rep, _n_test):
            total = None
            for name in train_g.data:
                b = train_g.data[name].float() @ rep[name].float().T
                total = b if total is None else total + b
            return total

        def run(batch):
            args = _args(tmp_path / "o")
            args.per_device_train_batch_size = batch  # the scoring batch
            train = DiskGradientSource(GradientStorageManager(train_dir), args)
            test = DiskGradientSource(GradientStorageManager(test_dir), args)
            return score_sources(
                train,
                test,
                args.device,
                prepare_test=prep,
                score_block=score_block,
            )

        s1, ids1, steps1, tids1 = run(1)  # one stored (3-doc) block per batch
        s5, ids5, steps5, _ = run(5)  # 5-doc batches (regroups 3-doc blocks)
        sn, idsn, stepsn, tidsn = run(100)  # whole 12-doc store as one batch
        assert (ids1, steps1, tids1) == (idsn, stepsn, tidsn)
        assert (ids1, steps1) == (ids5, steps5)

        torch.testing.assert_close(s1, s5)
        torch.testing.assert_close(s1, sn)

    def test_factorized_scored_per_block(self, tmp_path):
        # A factorized store can't be stacked into a dense (B, D) batch;
        # score_sources scores it one block at a time (no crash, right row count).
        from dattri_llm.attribution.utils import score_sources
        from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
        from dattri_llm.gradient.storage_manager import GradientStorageManager
        from dattri_llm.gradient.streaming import DiskGradientSource

        torch.manual_seed(0)
        fm = GradientStorageManager(str(tmp_path / "fac"))
        for s in range(3):
            g = Gradient(
                representation={"L0": "factorized"},
                data={"L0": Factorized(torch.randn(2, 4, 5), torch.randn(2, 4, 5))},
                layer_types={"L0": "nn.Linear"},
                validate_on_init=False,
            )
            fm.save_bulk(
                [GradientRecord(step=s, input_hash=[f"t{s}_0", f"t{s}_1"], gradient=g)],
            )
        args = _args(tmp_path / "o")
        train = DiskGradientSource(GradientStorageManager(str(tmp_path / "fac")), args)
        test = DiskGradientSource(GradientStorageManager(str(tmp_path / "fac")), args)
        scores, ids, _steps, _tids = score_sources(
            train,
            test,
            args.device,
            prepare_test=lambda g: g,
            score_block=lambda _t, _r, n: torch.zeros(2, n),
        )
        assert scores.shape[0] == len(ids) == 6
