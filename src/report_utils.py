from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "run_pipeline.sh").exists() or (cwd / "reports").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "run_pipeline.sh").exists():
        return cwd.parent
    return cwd


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "auto"


def save_notebook_artifacts(globals_dict: Dict[str, Any]) -> None:
    """Save figures and summary tables from a notebook execution.

    Uses figure titles when available to generate snake_case filenames and
    writes outputs into reports/figures and reports/tables under the repo root.
    """
    results_dir = _repo_root() / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save figures with titles when available
    if plt is not None:
        for i, num in enumerate(plt.get_fignums(), start=1):
            fig = plt.figure(num)
            title = None
            if fig._suptitle is not None:
                title = fig._suptitle.get_text()
            if not title:
                for ax in fig.get_axes():
                    ax_title = ax.get_title()
                    if ax_title:
                        title = ax_title
                        break
            name_part = _slugify(title or f"figure_{i}")
            fig_path = figures_dir / f"fig{i:02d}_{name_part}.png"
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Save summary tables
    if pd is not None:
        table_candidates = []
        for var_name, value in list(globals_dict.items()):
            if not isinstance(value, pd.DataFrame):
                continue
            # Heuristic: keep summary-sized tables and common artifact names
            if value.shape[0] > 5000:
                continue
            if value.shape[1] > 200:
                continue
            name_l = var_name.lower()
            if any(k in name_l for k in [
                "summary", "metric", "metrics", "kpi", "performance",
                "comparison", "confusion", "importance", "results", "report"
            ]) or value.shape[0] <= 200:
                table_candidates.append((var_name, value))

        for i, (var_name, df) in enumerate(table_candidates, start=1):
            name_part = _slugify(var_name)
            table_path = tables_dir / f"tab{i:02d}_{name_part}.csv"
            df.to_csv(table_path, index=False)
