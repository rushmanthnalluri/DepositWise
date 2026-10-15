"""Create a lightweight profile of the local bank marketing CSV."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.train import DATA_PATH, ROOT, read_dataset

REPORT_DIR = ROOT / "reports/generated"
FIGURE_DIR = ROOT / "reports/figures"


def main() -> None:
    frame = read_dataset(DATA_PATH)
    target_rate = frame["y"].value_counts(normalize=True).to_dict()
    missing = frame.isna().sum().sort_values(ascending=False)
    report = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "target_counts": {str(k): int(v) for k, v in frame["y"].value_counts().items()},
        "target_rate": {str(k): float(v) for k, v in target_rate.items()},
        "missing_values": {str(k): int(v) for k, v in missing.items() if v > 0},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duration_warning": "Duration is excluded from pre-call targeting because it is known only after contact.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    counts = frame["y"].value_counts().reindex(["no", "yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Term-deposit subscription target distribution")
    ax.set_xlabel("Subscription")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Profile saved to {REPORT_DIR / 'data_profile.json'}")
    print(f"Chart saved to {FIGURE_DIR / 'target_distribution.png'}")


if __name__ == "__main__":
    main()
