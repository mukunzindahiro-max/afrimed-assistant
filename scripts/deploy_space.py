import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import HfApi

from afrimed.config import MODELS_DIR
from afrimed.experiments import get_knowledge_index

SPACE_README = """---
title: AfriMed Assistant
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

# AfriMed Assistant

Medical question answering for African healthcare contexts. The model is fine-tuned from Qwen2.5-Instruct with LoRA on
[AfriMed-QA v2](https://huggingface.co/datasets/intronhealth/afrimedqa_v2) (CC BY-SA 4.0, Olatunji et al., 2024).
The retrieval index in `artifacts/` contains AfriMed-QA training QA pairs and is shared under the same CC BY-SA 4.0 licence.

Model: [{model_repo}](https://huggingface.co/{model_repo})
"""

SPACE_REQUIREMENTS = """torch
transformers==4.46.3
peft==0.13.2
accelerate==1.1.1
sentence-transformers==3.2.1
scikit-learn
joblib
pandas
numpy
rouge_score
sacrebleu
pyyaml
"""


def main():
    import gradio

    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="e.g. username/afrimed-assistant")
    parser.add_argument("--model-repo", required=True)
    args = parser.parse_args()

    build = ROOT / "build" / "space"
    if build.exists():
        shutil.rmtree(build)
    (build / "artifacts").mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", build / "app.py")
    shutil.copytree(ROOT / "src" / "afrimed", build / "afrimed", ignore=shutil.ignore_patterns("__pycache__"))
    get_knowledge_index()
    shutil.copytree(MODELS_DIR / "knowledge_index", build / "artifacts" / "knowledge_index")
    shutil.copytree(MODELS_DIR / "ood", build / "artifacts" / "ood")
    (build / "requirements.txt").write_text(SPACE_REQUIREMENTS)
    (build / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__,
                                                         model_repo=args.model_repo))
    app = (build / "app.py").read_text().replace(
        '"mukunzindahiro-max/afrimed-assistant-qwen2.5-0.5b-lora"', f'"{args.model_repo}"')
    (build / "app.py").write_text(app)

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(build), repo_id=args.space, repo_type="space")
    print(f"deployed https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
