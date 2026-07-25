"""OffloadCallback gradient-accumulation windows (store-then-attribute #19).

With ``gradient_accumulation_steps=N``, every window of N capture records is
merged into one stored record whose ``step`` is the optimizer-step counter --
parameters do not change within a window, so the merged record is exactly a
large-batch step.  Downstream readers see optimizer-step indexing with no
changes of their own.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from dattri_llm.gradient import ops
from dattri_llm.gradient.callbacks import OffloadCallback
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord
from dattri_llm.gradient.hooks import REGISTER_ALL, HookManager, HookManagerConfig
from dattri_llm.gradient.storage_manager import GradientStorageManager

B, IN_DIM, OUT_DIM = 2, 4, 3


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(IN_DIM, 8), nn.ReLU(), nn.Linear(8, OUT_DIM))


def _collect_to_disk(save_dir, batches, accum: int) -> GradientStorageManager:
    model = _model()
    fm = GradientStorageManager(str(save_dir))
    cb = OffloadCallback(
        offload_interval=1,
        file_manager=fm,
        gradient_accumulation_steps=accum,
    )
    hm = HookManager(
        model,
        config=HookManagerConfig(linear_io=REGISTER_ALL),
        callbacks=[cb],
    )
    try:
        with hm.collect():
            for x in batches:
                model.zero_grad()
                model(x).pow(2).sum().backward()
    finally:
        hm.remove()
    return GradientStorageManager(str(save_dir))  # fresh reader


class TestWindowMerging:
    def test_steps_are_update_indices_and_content_matches(self, tmp_path):
        """5 micro-batches, accum=2 -> stored steps [0, 1, 2] (2 + 2 + ragged
        1 micro-batches); each merged record's per-sample gradients equal the
        unmerged store's, in micro-batch order.
        """
        gen = torch.Generator().manual_seed(1)
        batches = [torch.randn(B, IN_DIM, generator=gen) for _ in range(5)]

        fm_micro = _collect_to_disk(tmp_path / "micro", batches, accum=1)
        fm_acc = _collect_to_disk(tmp_path / "acc", batches, accum=2)

        assert fm_micro.available_steps() == [0, 1, 2, 3, 4]
        assert fm_acc.available_steps() == [0, 1, 2]

        windows = [(0, [0, 1]), (1, [2, 3]), (2, [4])]
        for update_step, micro_steps in windows:
            ((file_rel, by_step),) = fm_acc.iter_steps(update_step)
            merged = fm_acc.load_records(file_rel)[by_step[update_step][0]]
            assert merged.step == update_step
            micro_records = []
            for s in micro_steps:
                ((f, micro_by_step),) = fm_micro.iter_steps(s)
                micro_records.append(fm_micro.load_records(f)[micro_by_step[s][0]])
            # Identifiers concatenate in micro-batch order.
            expected_hashes = [h for r in micro_records for h in r.input_hash]
            assert merged.input_hash == expected_hashes
            # Per-sample gradient content is preserved row-for-row.
            for name in micro_records[0].gradient.layer_names:
                m_merged = ops.materialize(merged.gradient.data[name], "nn.Linear")
                m_micro = torch.cat(
                    [
                        ops.materialize(r.gradient.data[name], "nn.Linear")
                        for r in micro_records
                    ],
                )
                assert torch.allclose(m_merged, m_micro, atol=1e-6), name

    def test_variable_length_micro_batches_merge(self, tmp_path):
        """Micro-batches with differing token lengths merge via zero-padding."""
        gen = torch.Generator().manual_seed(2)

        def rec(step, t, tag):
            return GradientRecord(
                step=step,
                input_hash=[f"{tag}{i}" for i in range(B)],
                gradient=Gradient(
                    representation={"l1": "factorized"},
                    data={
                        "l1": Factorized(
                            activation=torch.randn(B, t, IN_DIM, generator=gen),
                            pre_activation_grad=torch.randn(
                                B,
                                t,
                                OUT_DIM,
                                generator=gen,
                            ),
                        ),
                    },
                    layer_types={"l1": "nn.Linear"},
                    indexing={"l1": "batch_token"},
                ),
            )

        fm = GradientStorageManager(str(tmp_path))
        cb = OffloadCallback(
            offload_interval=1,
            file_manager=fm,
            gradient_accumulation_steps=2,
        )
        cb.on_step_end(rec(0, t=3, tag="a"))
        cb.on_step_end(rec(1, t=5, tag="b"))
        cb.on_context_end()

        reader = GradientStorageManager(str(tmp_path))
        ((file_rel, by_step),) = reader.iter_steps(0)
        merged = reader.load_records(file_rel)[by_step[0][0]]
        assert merged.input_hash == ["a0", "a1", "b0", "b1"]
        f = merged.gradient.data["l1"]
        assert f.activation.shape == (2 * B, 5, IN_DIM)  # padded to max T
        assert (f.pre_activation_grad[:B, 3:] == 0).all()

    def test_per_sample_recording_uses_update_steps(self, tmp_path):
        gen = torch.Generator().manual_seed(3)
        model = _model()
        fm = GradientStorageManager(str(tmp_path))
        cb = OffloadCallback(
            offload_interval=1,
            file_manager=fm,
            recording_type="per_sample",
            gradient_accumulation_steps=2,
        )
        hm = HookManager(
            model,
            config=HookManagerConfig(linear_io=REGISTER_ALL),
            callbacks=[cb],
        )
        try:
            with hm.collect():
                for _ in range(4):
                    model.zero_grad()
                    model(torch.randn(B, IN_DIM, generator=gen)).sum().backward()
        finally:
            hm.remove()

        reader = GradientStorageManager(str(tmp_path))
        assert reader.available_steps() == [0, 1]
        for step in (0, 1):
            slots = reader.iter_steps(step)
            n = sum(len(by_step[step]) for _, by_step in slots)
            assert n == 2 * B  # one per sample of the merged window


class TestStoreConvention:
    def test_metadata_recorded_and_readable(self, tmp_path):
        gen = torch.Generator().manual_seed(4)
        _collect_to_disk(
            tmp_path,
            [torch.randn(B, IN_DIM, generator=gen) for _ in range(2)],
            accum=2,
        )
        meta = json.loads((tmp_path / "index_meta.json").read_text())
        assert meta["gradient_accumulation_steps"] == 2
        assert GradientStorageManager(str(tmp_path)).gradient_accumulation_steps == 2

    def test_mixed_conventions_rejected(self, tmp_path):
        gen = torch.Generator().manual_seed(5)
        _collect_to_disk(
            tmp_path,
            [torch.randn(B, IN_DIM, generator=gen) for _ in range(2)],
            accum=2,
        )
        fm = GradientStorageManager(str(tmp_path))
        with pytest.raises(ValueError, match="cannot mix"):
            OffloadCallback(
                offload_interval=1,
                file_manager=fm,
                gradient_accumulation_steps=4,
            )

    def test_invalid_accumulation_raises(self, tmp_path):
        with pytest.raises(ValueError, match=">= 1"):
            OffloadCallback(
                offload_interval=1,
                file_manager=GradientStorageManager(str(tmp_path)),
                gradient_accumulation_steps=0,
            )
