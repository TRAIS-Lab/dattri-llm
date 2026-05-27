"""Data selection on GPT-2 (124 M) with a diverse training corpus.

tiny-gpt2 (2 layers, 64 hidden) is too small to show meaningful influence
score spread: ALL inputs push the model in almost the same gradient direction,
regardless of content (cosine sim ≈ 0.98-1.00 for everything).

GPT-2 (124 M, 12 layers, 768 hidden) is a properly-trained model.  Different
domains push different parameter sub-spaces, so influence scores span both
positive and negative values depending on how well each training sample aligns
with the validation target.

Experiment layout
-----------------
  Val target (the "goal"):
      Two sentences about NLP and machine-learning research.

  Training batch (diverse domains):
      1. NLP / ML prose      ← should align with val → POSITIVE
      2. Python code         ← syntax structure differs → near zero / mixed
      3. French text         ← different language statistics → lower / negative
      4. Mathematics         ← symbolic, no natural-language syntax → lower
      5. Cooking recipe      ← semantics completely unrelated → negative

  For each sample we print:
      * per-sample ghost influence score  (fast)
      * per-sample materialized score     (verification)
      * ground-truth cosine similarity of the full param.grad to val grad
      * DROP / keep decision (bottom-fraction mode)

Note: one forward+backward on CPU takes ~5 s for gpt2.  Allow ~60 s total.

Run:
    python examples/example_gpt2_data_selection.py
"""

from __future__ import annotations

import textwrap
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dattri_llm.gradient.callbacks import DataSelectionCallback
from dattri_llm.gradient.gradient import GradientRecord
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "gpt2"   # 124 M parameters — large enough for meaningful score spread
MAX_LEN  = 32       # keep sequences short for CPU speed

print(f"Loading {MODEL_ID} …")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s  "
      f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

hook_cfg = HookManagerConfig(hook_types={"mlp_io": None})

SEP  = "=" * 74
SSEP = "-" * 74


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

# Validation target — NLP / ML research prose.
VAL_TEXTS = [
    "Training data attribution identifies which examples most influence model predictions.",
    "Influence functions measure the effect of individual training points on model loss.",
]

# Training corpus — deliberately diverse domains.
TRAIN_ITEMS = [
    # ① NLP / ML — same domain as val
    ("NLP/ML",   "Gradient-based data selection removes low-quality training examples."),
    ("NLP/ML",   "Per-sample influence scores guide curriculum learning strategies."),
    ("NLP/ML",   "Transformer models generalise better when harmful data is removed."),

    # ② Python code — same tokenizer, very different n-gram statistics
    ("CODE",     "def softmax(x):\n    e = torch.exp(x); return e / e.sum(-1, keepdim=True)"),
    ("CODE",     "for epoch in range(n_epochs):\n    optimizer.zero_grad(); loss.backward()"),

    # ③ French — different language, entirely different vocabulary distribution
    ("FRENCH",   "Les modèles de langage apprennent à partir de grandes quantités de texte."),
    ("FRENCH",   "La sélection de données améliore la qualité des modèles entraînés."),

    # ④ Mathematics — symbolic notation, no linguistic structure
    ("MATH",     "Let f(x) = Σ_i w_i φ_i(x). The gradient ∂L/∂w_i = ∂L/∂f · φ_i(x)."),
    ("MATH",     "argmin_θ Σ_i L(f_θ(x_i), y_i) + λ ||θ||²  subject to θ ∈ Θ."),

    # ⑤ Cooking — completely unrelated semantics
    ("COOKING",  "Preheat the oven to 375°F. Mix flour, sugar, and butter until crumbly."),
    ("COOKING",  "Simmer the sauce for 20 minutes, stirring occasionally, until thickened."),
]

TRAIN_TAGS  = [tag  for tag,  _    in TRAIN_ITEMS]
TRAIN_TEXTS = [text for _,    text in TRAIN_ITEMS]


def encode(texts: list[str]) -> torch.Tensor:
    return tokenizer(
        texts, truncation=True, max_length=MAX_LEN,
        padding="max_length", return_tensors="pt",
    )["input_ids"]


def causal_lm_loss(m, batch: dict) -> torch.Tensor:
    ids = batch["input_ids"]
    return m(input_ids=ids, labels=ids).loss


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Capture val gradient (fixed target)
# ─────────────────────────────────────────────────────────────────────────────

class _Cap:
    def __init__(self): self.gradient = None
    def on_step_end(self, r: GradientRecord): self.gradient = r.gradient
    def on_layer_forward(self, *_): pass
    def on_layer_backward(self, *_): pass
    def on_context_end(self): pass


print(f"\nCapturing val gradient … ", end="", flush=True)
t0 = time.time()
cap = _Cap()
hm_cap = HookManager(model, config=hook_cfg, callbacks=[cap])
val_ids = encode(VAL_TEXTS)
with hm_cap.collect():
    model.zero_grad()
    causal_lm_loss(model, {"input_ids": val_ids}).backward()
hm_cap.remove()
val_gradient = cap.gradient
print(f"done ({time.time()-t0:.1f}s)")

# Also capture the full param.grad for cosine-similarity ground truth.
val_grad_flat = torch.cat([
    p.grad.detach().flatten()
    for p in model.parameters() if p.grad is not None
])

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Per-sample ground-truth cosine similarities
# ─────────────────────────────────────────────────────────────────────────────

print("Computing per-sample cosine similarities (ground truth) … ", end="", flush=True)
t0 = time.time()
cos_sims: list[float] = []
for text in TRAIN_TEXTS:
    model.zero_grad()
    causal_lm_loss(model, {"input_ids": encode([text])}).backward()
    flat = torch.cat([
        p.grad.detach().flatten()
        for p in model.parameters() if p.grad is not None
    ])
    cos_sims.append(F.cosine_similarity(flat, val_grad_flat, dim=0).item())
print(f"done ({time.time()-t0:.1f}s)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Ghost + materialized influence scores on the whole batch
# ─────────────────────────────────────────────────────────────────────────────

DROP_FRACTION = 0.35  # drop bottom 35 % of batch

ghost_cb = DataSelectionCallback(
    model=model, threshold=DROP_FRACTION,
    threshold_mode="bottom_fraction",
    target="fixed", target_gradient=val_gradient,
    score_mode="ghost",
)
mat_cb = DataSelectionCallback(
    model=model, threshold=DROP_FRACTION,
    threshold_mode="bottom_fraction",
    target="fixed", target_gradient=val_gradient,
    score_mode="materialized",
)

print("Running batched forward+backward for ghost/materialized scores … ", end="", flush=True)
t0 = time.time()
train_ids = encode(TRAIN_TEXTS)
hm = HookManager(model, config=hook_cfg, callbacks=[ghost_cb, mat_cb])
with hm.collect():
    model.zero_grad()
    causal_lm_loss(model, {"input_ids": train_ids}).backward()
hm.remove()
print(f"done ({time.time()-t0:.1f}s)")

ghost_scores = ghost_cb.last_scores
mat_scores   = mat_cb.last_scores
dropped_idx  = set(ghost_cb.last_dropped)

# ─────────────────────────────────────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  GPT-2 (124M) — diverse corpus influence scores")
print(SEP)

print(f"\n  Val target ({len(VAL_TEXTS)} sentence(s)):")
for vt in VAL_TEXTS:
    print(f"    \"{textwrap.shorten(vt, 70)}\"")

print(f"\n  Drop threshold: bottom {int(DROP_FRACTION*100)}% of batch "
      f"({len(dropped_idx)}/{len(TRAIN_TEXTS)} samples)")

print(f"\n  {'Dec':>4}  {'Tag':<8}  {'ghost':>9}  {'mat':>9}  {'cos_sim':>9}  Sample")
print(f"  {'----':>4}  {'--------':<8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*42}")

for i, (tag, text) in enumerate(TRAIN_ITEMS):
    dec   = "DROP" if i in dropped_idx else "keep"
    gs    = ghost_scores[i].item()
    ms    = mat_scores[i].item()
    cs    = cos_sims[i]
    short = textwrap.shorten(text.replace("\n", " ↵ "), width=42, placeholder="…")
    print(f"  {dec:>4}  {tag:<8}  {gs:+9.5f}  {ms:+9.5f}  {cs:+9.6f}  \"{short}\"")

print()

# Summary by domain
print(f"  {'Domain':<10}  {'mean ghost':>12}  {'mean cos_sim':>14}")
print(f"  {'----------':<10}  {'-'*12}  {'-'*14}")
from collections import defaultdict
domain_ghost: dict[str, list] = defaultdict(list)
domain_cos:   dict[str, list] = defaultdict(list)
for tag, gs, cs in zip(TRAIN_TAGS, ghost_scores.tolist(), cos_sims):
    domain_ghost[tag].append(gs)
    domain_cos[tag].append(cs)

for tag in ["NLP/ML", "CODE", "FRENCH", "MATH", "COOKING"]:
    mg = sum(domain_ghost[tag]) / len(domain_ghost[tag])
    mc = sum(domain_cos[tag])   / len(domain_cos[tag])
    print(f"  {tag:<10}  {mg:+12.5f}  {mc:+14.6f}")

print()
n_neg = (ghost_scores < 0).sum().item()
print(f"  Negative scores : {n_neg}/{len(TRAIN_TEXTS)}")
print(f"  Score range     : [{ghost_scores.min().item():+.5f}, "
      f"{ghost_scores.max().item():+.5f}]")
print(f"  ghost==mat diff : {(ghost_scores - mat_scores).abs().max().item():.2e}")
print(SSEP)
