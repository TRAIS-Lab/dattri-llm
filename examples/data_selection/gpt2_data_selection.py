"""This example shows online data selection on GPT-2 with a diverse training corpus.

A validation gradient (NLP/ML prose) is the selection target.  One batched
forward+backward over a mixed-domain training batch then scores every sample by
gradient alignment with that target (the factorized "ghost" inner product) and
drops the bottom fraction, exactly as DataSelectionCallback would inside a real
training loop.

Note: one forward+backward on CPU takes ~5 s for gpt2 (124M). Allow ~60 s total.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from collections import defaultdict

# Make the repo importable when running the script directly (no install needed).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dattri_llm.gradient.callbacks import CaptureCallback, DataSelectionCallback
from dattri_llm.gradient.hooks import HookManager, HookManagerConfig, REGISTER_ALL

MODEL_ID = "gpt2"   # 124M parameters -- large enough for meaningful score spread
MAX_LEN = 32        # keep sequences short for CPU speed

# validation target -- NLP / ML research prose (the "goal")
VAL_TEXTS = [
    "Training data attribution identifies which examples most influence model predictions.",
    "Influence functions measure the effect of individual training points on model loss.",
]

# training corpus -- deliberately diverse domains
TRAIN_ITEMS = [
    ("NLP/ML",  "Gradient-based data selection removes low-quality training examples."),
    ("NLP/ML",  "Per-sample influence scores guide curriculum learning strategies."),
    ("NLP/ML",  "Transformer models generalise better when harmful data is removed."),
    ("CODE",    "def softmax(x):\n    e = torch.exp(x); return e / e.sum(-1, keepdim=True)"),
    ("CODE",    "for epoch in range(n_epochs):\n    optimizer.zero_grad(); loss.backward()"),
    ("FRENCH",  "Les modèles de langage apprennent à partir de grandes quantités de texte."),
    ("FRENCH",  "La sélection de données améliore la qualité des modèles entraînés."),
    ("MATH",    "Let f(x) = Σ_i w_i φ_i(x). The gradient ∂L/∂w_i = ∂L/∂f · φ_i(x)."),
    ("MATH",    "argmin_θ Σ_i L(f_θ(x_i), y_i) + λ ||θ||²  subject to θ ∈ Θ."),
    ("COOKING", "Preheat the oven to 375°F. Mix flour, sugar, and butter until crumbly."),
    ("COOKING", "Simmer the sauce for 20 minutes, stirring occasionally, until thickened."),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_fraction", type=float, default=0.35,
                        help="Fraction of the batch to drop (bottom scores).")
    args_cli = parser.parse_args()

    # load the pretrained model
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()

    def encode(texts):
        return tokenizer(texts, truncation=True, max_length=MAX_LEN,
                         padding="max_length", return_tensors="pt")["input_ids"]

    def causal_lm_loss(m, ids):
        return m(input_ids=ids, labels=ids).loss

    hook_cfg = HookManagerConfig(linear_io=REGISTER_ALL)
    train_tags = [tag for tag, _ in TRAIN_ITEMS]
    train_texts = [text for _, text in TRAIN_ITEMS]

    # step 1 -- capture the validation gradient (the fixed selection target)
    print("Capturing the validation-target gradient ...")
    cap = CaptureCallback()
    hm = HookManager(model, config=hook_cfg, callbacks=[cap])
    with hm.collect():
        model.zero_grad()
        causal_lm_loss(model, encode(VAL_TEXTS)).backward()
    hm.remove()
    val_gradient = cap.record.gradient

    # keep the raw param.grad too, as a ground-truth reference direction
    val_grad_flat = torch.cat([p.grad.detach().flatten()
                               for p in model.parameters() if p.grad is not None])

    # step 2 -- ground truth: per-sample cosine similarity of the full
    # param.grad against the validation gradient (one backward per sample)
    print("Computing per-sample cosine similarities (ground truth) ...")
    cos_sims = []
    for text in train_texts:
        model.zero_grad()
        causal_lm_loss(model, encode([text])).backward()
        flat = torch.cat([p.grad.detach().flatten()
                          for p in model.parameters() if p.grad is not None])
        cos_sims.append(F.cosine_similarity(flat, val_grad_flat, dim=0).item())

    # step 3 -- one batched forward+backward scores every sample at once.
    # DataSelectionCallback computes <dL_i/dW, dL_val/dW> per sample from the
    # factorized ("ghost") gradients and drops the bottom fraction.
    print("Scoring the training batch ...")
    ghost_cb = DataSelectionCallback(
        model=model, threshold=args_cli.drop_fraction,
        threshold_mode="bottom_fraction",
        target="fixed", target_gradient=val_gradient,
        score_mode="ghost",
    )
    hm = HookManager(model, config=hook_cfg, callbacks=[ghost_cb])
    with hm.collect():
        model.zero_grad()
        causal_lm_loss(model, encode(train_texts)).backward()
    hm.remove()

    ghost_scores = ghost_cb.last_scores
    dropped_idx = set(ghost_cb.last_dropped)

    # the selection target and the per-sample results
    print(f"\nValidation target ({len(VAL_TEXTS)} sentences):")
    for vt in VAL_TEXTS:
        print(f"  \"{textwrap.shorten(vt, 80)}\"")
    print(f"\nDrop threshold: bottom {int(args_cli.drop_fraction * 100)}% of batch "
          f"({len(dropped_idx)}/{len(train_texts)} samples)")
    print(f"\n{'Dec':<6}{'Tag':<9}{'score':>13}{'cos_sim':>11}  Sample")
    print("-" * 87)
    for i, (tag, text) in enumerate(TRAIN_ITEMS):
        dec = "DROP" if i in dropped_idx else "keep"
        short = textwrap.shorten(text.replace("\n", " "), width=42, placeholder="...")
        print(f"{dec:<6}{tag:<9}{ghost_scores[i].item():>+13.4f}"
              f"{cos_sims[i]:>+11.6f}  \"{short}\"")
    print("-" * 87)

    # summary by domain -- same-domain samples should score highest
    domain_ghost, domain_cos = defaultdict(list), defaultdict(list)
    for tag, gs, cs in zip(train_tags, ghost_scores.tolist(), cos_sims):
        domain_ghost[tag].append(gs)
        domain_cos[tag].append(cs)
    print(f"\n{'Domain':<10}{'mean score':>14}{'mean cos_sim':>15}")
    print("-" * 39)
    for tag in ["NLP/ML", "CODE", "FRENCH", "MATH", "COOKING"]:
        mg = sum(domain_ghost[tag]) / len(domain_ghost[tag])
        mc = sum(domain_cos[tag]) / len(domain_cos[tag])
        print(f"{tag:<10}{mg:>+14.4f}{mc:>+15.6f}")
    print("-" * 39)
