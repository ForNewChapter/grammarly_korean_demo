#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

COMMANDS = [
    [PYTHON, str(BASE_DIR / 'scripts' / 'prepare_aihub_highfreq_dataset.py')],
    [PYTHON, str(BASE_DIR / 'scripts' / 'build_high_precision_surface_fixes.py')],
    [PYTHON, str(BASE_DIR / 'scripts' / 'build_typed_confusion_graph.py')],
    [PYTHON, str(BASE_DIR / 'scripts' / 'build_lemma_family_graph.py')],
    [PYTHON, str(BASE_DIR / 'scripts' / 'build_predicate_family_seeds.py')],
    [PYTHON, str(BASE_DIR / 'scripts' / 'build_phrase_memory.py')],
]


def main() -> None:
    for command in COMMANDS:
        print(f"[build_candidate_assets] running: {' '.join(command)}")
        subprocess.run(command, cwd=BASE_DIR, check=True)


if __name__ == '__main__':
    main()
