"""Read-path tests for on-disk gradient datasets (gradient/datasets.py).

Regression under test: per-sample records stored at one step with differing
sequence lengths (no padding-to-max collation) used to fail at *read* time --
``_records_to_block`` batch-concatenates the records' ``(B, T, d)`` factors,
and ``torch.cat`` requires equal ``T``.  ``Gradient.concatenate`` now pads the
shorter side's token axis with exactly-zero gradient rows, which every
downstream operation treats as inert.
"""

from __future__ import annotations

import torch

from dattri_llm.gradient import ops
from dattri_llm.gradient.datasets import GradientFileDataset
from dattri_llm.gradient.file_manager import GradientFileManager
from dattri_llm.gradient.gradient import Factorized, Gradient, GradientRecord

B1, T1 = 1, 4
B2, T2 = 1, 7
D_IN, D_OUT = 3, 5

HASH_A = "aaaaaaaa" * 8
HASH_B = "bbbbbbbb" * 8


def _record(step: int, input_hash: str, t: int, seed: int) -> GradientRecord:
    gen = torch.Generator().manual_seed(seed)
    gradient = Gradient(
        representation={"l1": "factorized"},
        data={
            "l1": Factorized(
                activation=torch.randn(1, t, D_IN, generator=gen),
                pre_activation_grad=torch.randn(1, t, D_OUT, generator=gen),
            ),
        },
        layer_types={"l1": "nn.Linear"},
        indexing={"l1": "batch_token"},
    )
    return GradientRecord(step=step, input_hash=[input_hash], gradient=gradient)


class TestVariableLengthBlocks:
    def test_block_from_records_with_differing_seq_lengths(self, tmp_path):
        fm = GradientFileManager(str(tmp_path))
        rec_a = _record(0, HASH_A, t=T1, seed=1)
        rec_b = _record(0, HASH_B, t=T2, seed=2)
        fm.save_bulk([rec_a, rec_b])

        # Fresh manager, as an attributor would open the store.
        ds = GradientFileDataset(GradientFileManager(str(tmp_path)), step=0)
        assert len(ds) == 1
        block, hashes = ds[0]

        assert hashes == [HASH_A, HASH_B]
        assert block.batch_size == 2
        assert block.token_dim == {"l1": T2}  # padded to the longer record

        # Padding is inert: each row's materialized gradient equals the
        # original record's.
        m = ops.materialize(block.data["l1"], "nn.Linear")
        m_a = ops.materialize(rec_a.gradient.data["l1"], "nn.Linear")
        m_b = ops.materialize(rec_b.gradient.data["l1"], "nn.Linear")
        assert torch.allclose(m[0], m_a[0], atol=1e-6)
        assert torch.allclose(m[1], m_b[0], atol=1e-6)
