import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afrimed.baselines import run_baselines
from afrimed.ood import run_ood_experiment


def main():
    for m in run_baselines(use_bertscore="--no-bertscore" not in sys.argv):
        print(m["name"], {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    if "--skip-ood" not in sys.argv:
        report = run_ood_experiment()
        for method in ("tfidf_knn", "embedding_knn", "embedding_logreg"):
            print(method, {k: round(v, 4) for k, v in report[method].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
