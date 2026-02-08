#!/usr/bin/env python3
"""Plot per-user F1 score distribution for the winning model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    perf_path = root / "results" / "tables" / "model_user_performance.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"Missing {perf_path}. Run modeling first.")

    df = pd.read_csv(perf_path)
    # Prefer Logit_L2
    model_name = "Logit_L2"
    model_df = df[df['Model'] == model_name].copy()
    if model_df.empty:
        raise ValueError(f"No rows found for model: {model_name}")

    out = root / "results" / "figures" / "user_performance_boxplot.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=model_df['Mean_User_Accuracy'], color="#94a3b8")
    plt.title("User Performance Distribution (Logit_L2)")
    plt.ylabel("Mean Accuracy")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
