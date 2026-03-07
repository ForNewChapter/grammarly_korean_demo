#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


DEFAULT_DATASET_DIR = Path("./data/edit_tagger_dataset")
DEFAULT_OUTPUT_DIR = Path("./models/edit_tagger_koelectra")
DEFAULT_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KoELECTRA 기반 EditTagger를 학습합니다.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_dataset_dict(dataset_dir: Path, train_limit: int, validation_limit: int) -> DatasetDict:
    train_rows = read_jsonl(dataset_dir / "train.jsonl")
    valid_rows = read_jsonl(dataset_dir / "validation.jsonl")

    if train_limit > 0:
        train_rows = train_rows[:train_limit]
    if validation_limit > 0:
        valid_rows = valid_rows[:validation_limit]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(valid_rows),
        }
    )


@dataclass
class LabelConfig:
    labels: List[str]
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def load_label_config(dataset_dir: Path) -> LabelConfig:
    with (dataset_dir / "label_config.json").open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return LabelConfig(
        labels=payload["labels"],
        label2id={key: int(value) for key, value in payload["label2id"].items()},
        id2label={int(key): value for key, value in payload["id2label"].items()},
    )


def tokenize_and_align(tokenizer, label2id: Dict[str, int], max_length: int):
    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        labels: List[int] = []
        previous_word_idx = None
        word_ids = tokenized.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(label2id[batch["labels"][word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx
        tokenized["labels"] = labels
        return tokenized

    return _tokenize


def compute_metrics_factory(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        total = 0
        correct = 0
        non_keep_gold = 0
        non_keep_pred = 0
        non_keep_tp = 0

        for pred_row, gold_row in zip(predictions, labels):
            for pred_id, gold_id in zip(pred_row, gold_row):
                if gold_id == -100:
                    continue
                total += 1
                if pred_id == gold_id:
                    correct += 1
                gold_label = id2label[int(gold_id)]
                pred_label = id2label[int(pred_id)]
                if gold_label != "KEEP":
                    non_keep_gold += 1
                if pred_label != "KEEP":
                    non_keep_pred += 1
                if gold_label != "KEEP" and pred_label == gold_label:
                    non_keep_tp += 1

        precision = non_keep_tp / non_keep_pred if non_keep_pred else 0.0
        recall = non_keep_tp / non_keep_gold if non_keep_gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        return {
            "accuracy": correct / total if total else 0.0,
            "non_keep_precision": precision,
            "non_keep_recall": recall,
            "non_keep_f1": f1,
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_config = load_label_config(dataset_dir)
    dataset = load_dataset_dict(dataset_dir, args.train_limit, args.validation_limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_config.labels),
        id2label=label_config.id2label,
        label2id=label_config.label2id,
    )

    tokenized = dataset.map(
        tokenize_and_align(tokenizer, label_config.label2id, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "non_keep_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": args.seed,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_signature.parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in training_signature.parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError("TrainingArguments가 evaluation/eval strategy 인자를 지원하지 않습니다.")

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": compute_metrics_factory(label_config.id2label),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))

    metrics = {
        "train": train_result.metrics,
        "eval": eval_result,
        "label_config": {
            "labels": label_config.labels,
            "label2id": label_config.label2id,
            "id2label": label_config.id2label,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
