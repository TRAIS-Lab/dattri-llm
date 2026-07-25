"""kronfluence side: EKFAC influence on GPT-2 / wikitext-2, following the
library's own ``examples/wikitext/analyze.py`` (fp32, single GPU).

Queries are the first 32 validation blocks (= one query batch of their
documented ``--query_batch_size 32``); train side is the full train split.

    python run_kronfluence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))              # common.py
sys.path.insert(0, "/scratch/shixuanl/kronfluence")  # examples package

import torch
from transformers import default_data_collator

from common import Meter, dir_bytes, env_info

from examples.wikitext.pipeline import construct_gpt2, get_wikitext_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs

sys.path.insert(0, str(HERE))
from task_def import N_QUERY, QUERY_BATCH, TRAIN_BATCH, LanguageModelingTask


def main():
    train_dataset = get_wikitext_dataset(split="eval_train")
    eval_dataset = get_wikitext_dataset(split="valid", indices=list(range(N_QUERY)))

    model = construct_gpt2()
    model.load_state_dict(
        torch.load(HERE / "checkpoints" / "model.pth", weights_only=True))

    task = LanguageModelingTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="wikitext_eff",
        model=model,
        task=task,
        output_dir=str(HERE / "kronfluence_store"),
    )
    analyzer.set_dataloader_kwargs(
        DataLoaderKwargs(collate_fn=default_data_collator))

    meter = Meter(f"kronfluence(ekfac,b{TRAIN_BATCH})", "gpt2-wikitext2-ekfac",
                  HERE / "results.jsonl",
                  meta=env_info({"train_batch": TRAIN_BATCH}))

    factor_args = FactorArguments(strategy="ekfac")
    with meter.phase("fit_factors", len(train_dataset)):
        analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=train_dataset,
            per_device_batch_size=TRAIN_BATCH,
            factor_args=factor_args,
            overwrite_output_dir=True,
        )
    with meter.phase("pairwise_scores", len(train_dataset) + N_QUERY):
        analyzer.compute_pairwise_scores(
            scores_name="ekfac",
            score_args=ScoreArguments(),
            factors_name="ekfac",
            query_dataset=eval_dataset,
            train_dataset=train_dataset,
            per_device_query_batch_size=QUERY_BATCH,
            per_device_train_batch_size=TRAIN_BATCH,
            overwrite_output_dir=True,
        )
    scores = analyzer.load_pairwise_scores("ekfac")["all_modules"]
    torch.save({"score": scores.T.cpu().float()},  # -> (n_train, n_query)
               HERE / "score_kronfluence.pt")
    meter.note(n_train=len(train_dataset), score_shape=list(scores.T.shape),
               store_bytes=dir_bytes(HERE / "kronfluence_store"))
    meter.flush()


if __name__ == "__main__":
    main()
