"""Mixture-of-Experts capture through ``HookManager(moe_routed_layers=...)``.

The experts and the router of an MoE block receive 2-D token matrices --
the block input flattened to ``(B*T, d)`` for the router, the ``(n_e, d)``
rows routed to each expert -- so read as-is they claim ``B*T`` or ``n_e``
samples.  These tests pin the guaranteed behaviour on tiny Hugging Face MoE
models in float64: an undeclared block fails with a message naming the
layers; a declared one gives exactly the per-sample gradient of every
layer (experts, router, shared expert, attention) that a per-sample autograd
loop gives, through the raw, projected and materialized capture paths,
under gradient checkpointing, across steps of different routing, and under
DDP.
"""

from __future__ import annotations

import os
import pathlib
import socket
import tempfile
import warnings

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from dattri_llm.gradient import ops
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

transformers = pytest.importorskip("transformers")
if hasattr(transformers.models.mixtral.modeling_mixtral, "MixtralExperts"):
    pytest.skip(
        "per-expert nn.Linear layout (Transformers 4); the fused layout is "
        "covered by test_fused_experts.py",
        allow_module_level=True,
    )

B, T, VOCAB = 3, 7, 64
# Experts and router of layer 0 of a 2-layer model; ``shared`` is Qwen2-MoE's
# always-on expert, which sees the flattened block input.
COMMON = {
    "vocab_size": VOCAB,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
    "attn_implementation": "eager",
}


def _build(family: str) -> tuple[torch.nn.Module, list[str]]:
    """A tiny MoE model of *family* and the names of its MoE blocks."""
    torch.manual_seed(0)
    if family == "mixtral":
        cfg = transformers.MixtralConfig(
            intermediate_size=32,
            num_local_experts=4,
            num_experts_per_tok=2,
            **COMMON,
        )
        model = transformers.MixtralForCausalLM(cfg)
        blocks = [f"model.layers.{i}.block_sparse_moe" for i in range(2)]
    elif family == "qwen2_moe":
        # moe_intermediate_size == hidden_size on purpose: an expert's down
        # projection then has block-input width without holding block rows.
        cfg = transformers.Qwen2MoeConfig(
            intermediate_size=32,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=24,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            **COMMON,
        )
        model = transformers.Qwen2MoeForCausalLM(cfg)
        blocks = [f"model.layers.{i}.mlp" for i in range(2)]
    elif family == "olmoe":
        cfg = transformers.OlmoeConfig(
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            **COMMON,
        )
        model = transformers.OlmoeForCausalLM(cfg)
        blocks = [f"model.layers.{i}.mlp" for i in range(2)]
    else:
        raise ValueError(family)
    return model.double(), blocks


def _batch(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T), generator=gen)


def _per_sample_loss(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Each sample's mean next-token loss, so the batch loss is their sum and
    the batched per-sample gradient equals the one-sample gradient.
    """
    logits = model(input_ids=ids).logits[:, :-1]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    return ce.view(ids.shape[0], -1).mean(1)


def _reference(model: torch.nn.Module, ids: torch.Tensor) -> list[dict]:
    """Per-sample parameter gradients from one backward per sample."""
    out = []
    for i in range(ids.shape[0]):
        model.zero_grad()
        _per_sample_loss(model, ids[i : i + 1]).sum().backward()
        out.append(
            {n: p.grad.detach().clone() for n, p in model.named_parameters()},
        )
    model.zero_grad()
    return out


def _collect(model, ids, **hm_kwargs):
    hm = HookManager(model, **hm_kwargs)
    try:
        with hm.collect():
            model.zero_grad()
            _per_sample_loss(model, ids).sum().backward()
        return hm.get_gradient()
    finally:
        hm.remove()


def _weight_block(dense_row: torch.Tensor, weight: torch.Tensor, has_bias: bool):
    """The weight part of one sample's materialized ``(d_out, d_in[+1])`` row."""
    if has_bias:
        return dense_row.reshape(weight.shape[0], -1)[:, : weight.shape[1]]
    return dense_row[: weight.numel()].reshape(weight.shape)


def _assert_matches_reference(model, gradient, reference, *, only=None):
    """Every factorized linear layer's materialized per-sample gradient equals
    the reference; returns the number of layers checked.
    """
    params = dict(model.named_parameters())
    checked = 0
    for name, value in gradient.data.items():
        if not hasattr(value, "activation"):  # factorized layers only
            continue
        if only is not None and not any(s in name for s in only):
            continue
        weight = params.get(name + ".weight")
        if weight is None or weight.ndim != 2:
            continue
        dense = ops.materialize(value, gradient.layer_types[name])
        has_bias = (name + ".bias") in params
        for i, ref in enumerate(reference):
            mine = _weight_block(dense[i], weight, has_bias)
            torch.testing.assert_close(
                mine, ref[name + ".weight"], rtol=1e-10, atol=1e-12
            )
        checked += 1
    return checked


# --------------------------------------------------------------------------- #
# Undeclared MoE block                                                          #
# --------------------------------------------------------------------------- #


class TestUndeclared:
    def test_batch_mismatch_names_router_and_option(self):
        model, _ = _build("mixtral")
        with pytest.raises(ValueError, match="same batch size") as exc:
            _collect(model, _batch())
        msg = str(exc.value)
        assert "block_sparse_moe.gate has 21" in msg  # B*T flattened rows
        assert "moe_routed_layers" in msg


# --------------------------------------------------------------------------- #
# Declared: exact per-sample gradients                                          #
# --------------------------------------------------------------------------- #


class TestRoutedCapture:
    @pytest.mark.parametrize("family", ["mixtral", "qwen2_moe", "olmoe"])
    def test_raw_capture_matches_per_sample_autograd(self, family):
        model, blocks = _build(family)
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(model, ids, moe_routed_layers=blocks)
        assert g.batch_size == B
        checked = _assert_matches_reference(model, g, reference)
        # Every 2-D weight of the model is checked: experts, router,
        # attention, embedding and head (and Qwen2-MoE's shared expert).
        n_matrices = sum(
            1
            for n, p in model.named_parameters()
            if n.endswith(".weight") and p.ndim == 2
        )
        assert checked == n_matrices

    def test_expert_layout_is_regrouped_by_sample(self):
        model, blocks = _build("mixtral")
        g = _collect(model, _batch(), moe_routed_layers=blocks)
        expert = "model.layers.0.block_sparse_moe.experts.0.w1"
        act = g.data[expert].activation
        assert act.ndim == 3
        assert act.shape[0] == B
        assert act.shape[1] <= T  # T_e: the most rows any sample routed here
        assert g.indexing[expert] == "routed_token"
        # Rows beyond a sample's count are exact zeros on both factors.
        grad = g.data[expert].pre_activation_grad
        counts = (act != 0).any(-1).sum(1)
        for i in range(B):
            assert torch.equal(
                grad[i, counts[i] :], torch.zeros_like(grad[i, counts[i] :])
            )
        # The router sees every token: the flattened input is reshaped back.
        router = "model.layers.0.block_sparse_moe.gate"
        assert tuple(g.data[router].activation.shape) == (B, T, 16)
        assert g.indexing[router] == "routed_token"

    def test_experts_and_router_may_be_declared_instead_of_the_block(self):
        model, _ = _build("mixtral")
        ids = _batch()
        reference = _reference(model, ids)
        declared = [f"model.layers.{i}.block_sparse_moe.experts" for i in range(2)]
        declared += [f"model.layers.{i}.block_sparse_moe.gate" for i in range(2)]
        g = _collect(model, ids, moe_routed_layers=declared)
        assert g.batch_size == B
        _assert_matches_reference(model, g, reference, only=["block_sparse_moe"])

    def test_declared_module_without_hooked_layers_warns(self):
        model, _ = _build("mixtral")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hm = HookManager(model, moe_routed_layers=["model.norm"])
            hm.remove()
        assert any("had no effect" in str(w.message) for w in caught)

    def test_unknown_module_name_raises(self):
        model, _ = _build("mixtral")
        with pytest.raises(ValueError, match="do not exist"):
            HookManager(model, moe_routed_layers=["model.layers.9.block_sparse_moe"])


# --------------------------------------------------------------------------- #
# Other capture paths                                                           #
# --------------------------------------------------------------------------- #


class TestCapturePaths:
    def test_logra_projection_sees_regrouped_rows(self):
        model, blocks = _build("mixtral")
        config = HookManagerConfig(
            projection_kwargs={
                "__default__": {"proj_dim": 8, "proj_max_batch_size": 8},
            },
        )
        g = _collect(model, _batch(), config=config, moe_routed_layers=blocks)
        expert = "model.layers.0.block_sparse_moe.experts.0.w1"
        assert g.batch_size == B
        act = g.data[expert].activation
        assert act.shape[0] == B
        assert act.shape[-1] == 8
        assert g.indexing[expert] == "routed_token"

    def test_materialized_style_matches_reference(self):
        model, blocks = _build("mixtral")
        ids = _batch()
        reference = _reference(model, ids)
        g = _collect(
            model,
            ids,
            config=HookManagerConfig(capture_style="materialized"),
            moe_routed_layers=blocks,
        )
        params = dict(model.named_parameters())
        for name, block in g.data.items():
            if (
                "block_sparse_moe" not in name
                or g.representation[name] != "materialized"
            ):
                continue
            weight = params[name + ".weight"]
            assert block.shape[0] == B
            for i, ref in enumerate(reference):
                torch.testing.assert_close(
                    block[i].reshape(weight.shape),
                    ref[name + ".weight"],
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_gradient_checkpointing_recompute_matches_reference(self):
        model, blocks = _build("mixtral")
        ids = _batch()
        reference = _reference(model, ids)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.train()
        try:
            g = _collect(model, ids, moe_routed_layers=blocks)
        finally:
            model.gradient_checkpointing_disable()
        assert g.batch_size == B
        _assert_matches_reference(model, g, reference, only=["block_sparse_moe"])


# --------------------------------------------------------------------------- #
# Downstream semantics                                                          #
# --------------------------------------------------------------------------- #


class TestRoutedTokenIndexing:
    def test_steps_with_different_routing_concatenate(self):
        model, blocks = _build("mixtral")
        hm = HookManager(model, moe_routed_layers=blocks)
        try:
            records = []
            for seed in (1, 2):
                with hm.collect():
                    model.zero_grad()
                    _per_sample_loss(model, _batch(seed)).sum().backward()
                records.append(hm.get_gradient())
        finally:
            hm.remove()
        expert = "model.layers.0.block_sparse_moe.experts.0.w1"
        t1 = records[0].data[expert].activation.shape[1]
        t2 = records[1].data[expert].activation.shape[1]
        cat = records[0].concatenate(records[1])
        assert cat.batch_size == 2 * B
        assert cat.data[expert].activation.shape[1] == max(t1, t2)
        # Padding is inert: the concatenated block materializes to the two
        # blocks stacked.
        torch.testing.assert_close(
            ops.materialize(cat.data[expert], "nn.Linear"),
            torch.cat([ops.materialize(r.data[expert], "nn.Linear") for r in records]),
        )

    def test_per_token_attribution_excludes_routed_layers(self):
        model, blocks = _build("mixtral")
        g = _collect(model, _batch(), moe_routed_layers=blocks)
        per_layer = g.similarity_per_token(g, reduce="none")
        assert per_layer
        assert not any("block_sparse_moe" in name for name in per_layer)
        expert = "model.layers.0.block_sparse_moe.experts.0.w1"
        with pytest.raises(ValueError, match="no token axis"):
            ops.layerwise_cross_dot_per_token(g, g, layers=[expert])

    def test_kfac_factors_ignore_padding_rows(self):
        model, blocks = _build("mixtral")
        g = _collect(model, _batch(), moe_routed_layers=blocks)
        expert = "model.layers.0.block_sparse_moe.experts.0.w1"
        f = g.data[expert]
        a_cov, g_cov = ops.kfac_factors(
            f.activation, f.pre_activation_grad, "nn.Linear"
        )
        rows = (f.pre_activation_grad != 0).any(-1)
        a_rows = f.activation[rows]  # the expert has no bias column
        g_rows = f.pre_activation_grad[rows]
        torch.testing.assert_close(a_cov, (a_rows.T @ a_rows / rows.sum()).float())
        torch.testing.assert_close(g_cov, (g_rows.T @ g_rows / rows.sum()).float())


# --------------------------------------------------------------------------- #
# DDP                                                                           #
# --------------------------------------------------------------------------- #


def _can_bind_localhost() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def _ddp_worker(rank, world_size, result_queue, rendezvous_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model, blocks = _build("mixtral")  # same seed -> same weights per rank
        ids = _batch(10 + rank)  # distinct per-rank batch
        reference = _reference(model, ids)
        hm = HookManager(model, moe_routed_layers=blocks)
        ddp = DDP(model)
        with hm.collect():
            ddp.zero_grad()
            logits = ddp(input_ids=ids).logits[:, :-1]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                reduction="none",
            )
            ce.view(B, -1).mean(1).sum().backward()
        g = hm.get_gradient()
        hm.remove()
        ok = g.batch_size == B
        report = f"rank {rank}: batch {g.batch_size}"
        try:
            n = _assert_matches_reference(
                model, g, reference, only=["block_sparse_moe"]
            )
            report += f", {n} MoE layers exact"
        except AssertionError as exc:  # pragma: no cover - diagnostic path
            ok = False
            report += f": {exc}"
        result_queue.put((rank, ok, report))
    finally:
        dist.destroy_process_group()


class TestRoutedCaptureDDP:
    def test_each_rank_captures_its_own_samples(self):
        if not _can_bind_localhost():
            pytest.skip("local socket binds are not permitted in this environment")
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        fd, rendezvous_path = tempfile.mkstemp()
        os.close(fd)
        try:
            mp.spawn(
                _ddp_worker,
                args=(2, result_queue, rendezvous_path),
                nprocs=2,
                join=True,
            )
        finally:
            if pathlib.Path(rendezvous_path).exists():
                pathlib.Path(rendezvous_path).unlink()
        results = [result_queue.get() for _ in range(2)]
        assert all(ok for _, ok, _ in results), results


class TestDuplicateRows:
    """Bit-identical token rows across samples (a repeated sample, a shared
    prefix under causal attention) are assigned to their own samples.
    """

    @pytest.mark.parametrize("kind", ["repeated_sample", "shared_prefix"])
    def test_expert_gradients_stay_exact(self, kind):
        model, blocks = _build("mixtral")
        base = _batch()
        if kind == "repeated_sample":
            ids = torch.stack([base[0], base[0], base[1]])
        else:
            ids = torch.stack([base[0], torch.cat([base[0, :4], base[1, 4:]]), base[2]])
        reference = _reference(model, ids)
        g = _collect(model, ids, moe_routed_layers=blocks)
        assert g.batch_size == B
        _assert_matches_reference(model, g, reference, only=["block_sparse_moe"])
