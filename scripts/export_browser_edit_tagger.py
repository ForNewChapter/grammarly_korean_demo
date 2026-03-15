#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="브라우저용 Edit Tagger 자산(config/tokenizer/onnx)을 생성합니다."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="학습된 HF 모델 디렉토리")
    parser.add_argument("--output-dir", type=Path, required=True, help="브라우저 자산 출력 디렉토리")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "q8"],
        default="fp32",
        help="브라우저에서 사용할 가중치 형식",
    )
    return parser.parse_args()


def export_to_onnx(input_dir: Path, export_dir: Path) -> Path:
    command = [
        sys.executable,
        "-m",
        "transformers.onnx",
        f"--model={input_dir}",
        "--feature=token-classification",
        "--framework=pt",
        "--export_with_transformers",
        str(export_dir),
    ]
    subprocess.run(command, check=True)
    return export_dir / "model.onnx"


def copy_base_assets(input_dir: Path, output_dir: Path, dtype: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy2(input_dir / name, output_dir / name)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["transformers.js_config"] = {
        "dtype": dtype,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_fp32_model(exported_onnx: Path, output_dir: Path) -> None:
    model = onnx.load(str(exported_onnx), load_external_data=True)
    target_dir = output_dir / "onnx"
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, target_dir / "model.onnx", save_as_external_data=False)


def write_q8_model(exported_onnx: Path, output_dir: Path) -> None:
    target_dir = output_dir / "onnx"
    target_dir.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(exported_onnx),
        str(target_dir / "model_quantized.onnx"),
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="edit_tagger_browser_export.") as temp_dir:
        export_dir = Path(temp_dir)
        exported_onnx = export_to_onnx(input_dir, export_dir)
        copy_base_assets(input_dir, output_dir, args.dtype)
        if args.dtype == "fp32":
            write_fp32_model(exported_onnx, output_dir)
        else:
            write_q8_model(exported_onnx, output_dir)

    print(f"browser_edit_tagger_output={output_dir}")


if __name__ == "__main__":
    main()
