#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 EditTagger를 ONNX로 export합니다.")
    parser.add_argument("--input-dir", type=Path, required=True, help="학습된 Hugging Face 모델 디렉토리")
    parser.add_argument("--output-dir", type=Path, required=True, help="ONNX export 출력 디렉토리")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(str(input_dir))
    model = ORTModelForTokenClassification.from_pretrained(str(input_dir), export=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print("export=done")


if __name__ == "__main__":
    main()
