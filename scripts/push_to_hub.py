import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from huggingface_hub import HfApi

from afrimed.config import MODELS_DIR, RESULTS_DIR
from afrimed.modeling import load_model, load_tokenizer
from afrimed.train import load_experiments

CARD = """---
license: apache-2.0
base_model: {base}
datasets:
- intronhealth/afrimedqa_v2
language:
- en
tags:
- medical
- africa
- lora
- question-answering
---

# AfriMed Assistant ({experiment})

`{base}` fine-tuned with LoRA on AfriMed-QA v2 (Olatunji et al., 2024) for African healthcare question answering.
LoRA adapters are merged into the base weights.

## Test results

{metrics}

This is a student research prototype for the ALU Machine Learning Techniques I summative project and is not intended for clinical use.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    cfg = load_experiments()[args.experiment]
    adapter = MODELS_DIR / args.experiment
    is_lora = (adapter / "adapter_config.json").exists()
    model = load_model(cfg.model_id if is_lora else str(adapter), cfg.kind,
                       adapter_path=str(adapter) if is_lora else None, dtype=torch.float32, device="cpu")
    tokenizer = load_tokenizer(str(adapter), cfg.kind)
    tokenizer.padding_side = "left" if cfg.kind == "causal" else "right"
    out = MODELS_DIR / f"{args.experiment}_merged"
    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    metrics_path = RESULTS_DIR / args.experiment / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    lines = "\n".join(f"- {k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float) and not k.endswith("_sec"))
    (out / "README.md").write_text(CARD.format(base=cfg.model_id, experiment=args.experiment, metrics=lines))

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model")
    print(f"pushed https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
