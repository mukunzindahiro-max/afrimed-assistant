import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afrimed.experiments import run_experiment
from afrimed.train import load_experiments


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AfriMed-QA experiments")
    parser.add_argument("--names", nargs="*", help="Experiment names from configs/experiments.yaml")
    parser.add_argument("--max-priority", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify the pipeline end-to-end")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics already exist")
    args = parser.parse_args()

    experiments = load_experiments()
    selected = [experiments[n] for n in args.names] if args.names else \
        sorted(experiments.values(), key=lambda c: c.priority)
    if args.max_priority is not None:
        selected = [c for c in selected if c.priority <= args.max_priority]

    for cfg in selected:
        if args.smoke:
            adapter = f"smoke_{cfg.adapter_from}" if cfg.adapter_from else None
            cfg = dataclasses.replace(cfg, name=f"smoke_{cfg.name}", adapter_from=adapter, eval_batch_size=4,
                                      gen_max_new_tokens=16, batch_size=2, grad_accum=1)
            run_experiment(cfg, limit=8, max_steps=2, skip_existing=False, use_bertscore=False)
        else:
            run_experiment(cfg, skip_existing=not args.force, use_bertscore=not args.no_bertscore)


if __name__ == "__main__":
    main()
