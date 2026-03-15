from pathlib import Path

from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    local_edit_tagger_dirs = [
        base_dir / "models" / "edit_tagger_v2_best" / "best",
        base_dir / "models" / "edit_tagger_v1_best" / "best",
    ]

    # Preload tokenizer/model artifacts into the image so runtime can stay offline.
    get_tokenizer()
    get_kobert_model().eval()
    AutoModelForMaskedLM.from_pretrained("monologg/kobert-lm").eval()
    AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").eval()

    for model_dir in local_edit_tagger_dirs:
        if model_dir.exists():
            AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
            AutoModelForTokenClassification.from_pretrained(str(model_dir)).eval()
            break


if __name__ == "__main__":
    main()
