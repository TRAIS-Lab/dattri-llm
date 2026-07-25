# Efficiency benchmark report: dattri_llm vs five TDA libraries

All numbers below are from the final SLURM re-run (jobs 4934-4938, one A40 +
120 GB cgroup memory cap per job) at the current library commit, which
includes the two fixes this benchmark surfaced (on-GPU capture default
``offload_to_cpu=False``, and the val-pass hook-state leak fix).  Pre-fix
measurements are archived per pair as ``results_pre_fix.jsonl``.

Environment: torch 2.11.0+cu130, transformers 4.55.4, fp32 (TF32 where a
side enables it), single NVIDIA A40 (44 GB).  Every pair's exact setting,
fairness notes, and reproduction command are in its own ``README.md``;
regenerate any pair with ``sbatch --job-name=eff-<pair> slurm/pair.sbatch <pair>``.
Largest per-experiment disk footprint: 21 GB (vs_bergson), well under the
500 GB cap.

## Headline table

| pair (their flagship setting) | ours | theirs | score agreement |
|---|---|---|---|
| **dattri** — mnist/mlp Grad-Dot, 5000x500 | **1.3 s, 0.03 GB, exact score** | 17.4 s as shipped (CPU projector) / 2.1 s best-config; always a proj-512 sketch, fidelity 0.89 | ours == autograd (1.0) |
| **logix** — GPT-2, wikitext-103 0.5%, LoGRA rank-64, 16 queries | otf **79 s** (no store) / disk 92 s (13.8 GB token-level store) | kfac 41 s (0.9 GB) / raw 48 s (3.9 GB) | **0.95** vs logix-kfac (their own raw-vs-kfac: 0.57) |
| **kronfluence** — GPT-2 (their fine-tune), wikitext-2, EKFAC full space, 4656x32 | **786 s, 41.7 GB, no store** (train batch 16) | 696 s, 38.8 GB, 2.7 GB store (batch 32) | **0.991** |
| **GhostSuite** — online train-vs-val grad-dots during GPT2-Small training | **2.08 s/step** (plain step 0.50 s) | 0.86 s/step compiled / 0.93 s eager | **1.000** (exact) |
| **bergson** — pythia-14m + pile-10k, rank-16 projected-gradient index | build 202 s, 1.9 GB GPU, 21 GB store; query 186 s | build 135 s, 5.2 GB (device-wide), 245 MB store; query 0.95 s | 0.432 == the rank-16 sketch-noise ceiling (ours reseeded vs itself: 0.438) |

## Reading guide

* **Correctness cross-validation.** Every pair checks that the two libraries
  compute the same quantity.  Ours matches plain autograd exactly (dattri
  pair), kronfluence's EKFAC at 0.991, GhostSuite's engines at 1.000, and
  logix-kfac at 0.95 despite each side using its own random sketch.  For
  bergson, agreement equals the information-theoretic ceiling of the rank-16
  sketch itself.
* **Where ours wins.** Exact (unprojected) grad-dot scoring is both faster
  and >30x more memory-efficient than dattri's mandatory sketch.  The online
  scoring callback computes *exact* per-sample dots at 2.08 s/step with zero
  training-loop changes.  OTF K-FAC/EK-FAC needs **no gradient store at
  all** (kronfluence persists 2.7 GB of factors; logix 0.9-3.9 GB;
  bergson 245 MB).
* **Where ours pays.** Our on-disk format keeps **per-token** projected
  factors -- the artifact that enables per-token attribution -- so stores are
  ~15-85x larger than sample-level stores (13.8 GB vs 0.9 GB on the logix
  pair; 21 GB vs 245 MB on the bergson pair) and disk-scoring passes are
  read-bound (bergson's in-memory 256-d index answers queries in ~1 s vs our
  186 s re-read).  GhostSuite's fused in-backward engine still holds a ~2.4x
  per-step edge over our hook-then-score design.
* **Benchmark-driven library fixes.**  The pre-fix runs exposed two real
  defects (archived in ``results_pre_fix.jsonl``): unprojected captures were
  copied to CPU every step (~15 GB/step of PCIe traffic), and the online
  val-gradient pass leaked ~20 GB of host RSS per step (repeated runs took
  the host down -- hence the SLURM-only policy).  With both fixed, the
  GhostSuite-pair step time fell 37.7 s -> 2.08 s (18x) and the kronfluence
  pair fell 3249 s -> 786 s (4.1x).

## Per-pair notes

* **vs_dattri** -- dattri's ``TracInAttributor`` cannot disable its random
  projection (``projector_kwargs=None`` still installs proj-512 on CPU), so
  its score is inherently a sketch; we report their shipped and
  best-configured (CUDA projector) variants.
* **vs_logix** -- projected space matched exactly (rank-64 -> 4096
  dims/layer on the same 48 GPT-2 linears).  Ours-disk vs ours-otf are
  bit-identical (1.0), and both agree with logix-kfac far more than logix's
  own raw and kfac modes agree with each other.
* **vs_kronfluence** -- same estimator, same damping (1e-8), same 48
  modules; ours ran train batch 16 (their 32) because the scoring pass holds
  all 32 queries' rotated test representations (~11 GB) alongside capture.
* **vs_ghostsuite** -- both sides train on identical seeded batches; the
  meaningful metric is overhead over the plain 0.50 s step.  Their compiled
  fast path pays a one-time 16-42 s torch.compile warmup.
* **vs_bergson** -- both build times are warm-cache; each side draws its own
  rank-16 sketch, so the 0.43 agreement was validated against an
  ours-vs-ours different-seed run (0.438): the libraries agree exactly as
  much as the sketch dimension permits.
