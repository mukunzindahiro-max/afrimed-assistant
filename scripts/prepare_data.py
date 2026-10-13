import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afrimed.data import prepare


def main():
    _, sup, unlabeled, report = prepare(use_cache="--refresh" not in sys.argv)
    print(json.dumps(report, indent=2))
    print(f"supervised examples: {len(sup)}, unlabeled consumer queries: {len(unlabeled)}")


if __name__ == "__main__":
    main()
