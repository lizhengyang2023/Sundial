"""Plot train/validation curves, or compare two zero-shot result JSONs.

Run from the repository root:
    python src/utils/draw_curves.py
    python src/utils/draw_curves.py --input-losses checkpoints/losses.json --output checkpoints/loss_curves.png
    python src/utils/draw_curves.py --zero-shot-a outputs/full_model/zero_shot.json \\
        --zero-shot-b outputs/original_model/zero_shot.json --output outputs/zero_shot_compare.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("flow_loss", "mae", "mse")
ZS_METRICS = ("mse", "mae")


def load_epoch_series(path: Path) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    """Return epoch ids and {split: {metric: values}} from losses.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no epoch records")
    rows = sorted(rows, key=lambda row: row["epoch"])
    epochs = [int(row["epoch"]) for row in rows]
    series: dict[str, dict[str, list[float]]] = {
        "train": {name: [] for name in METRICS},
        "validation": {name: [] for name in METRICS},
    }
    for row in rows:
        for split in series:
            block = row.get(split)
            if not isinstance(block, dict):
                raise ValueError(f"epoch {row.get('epoch')} missing '{split}' metrics")
            for name in METRICS:
                if name not in block:
                    raise ValueError(f"epoch {row.get('epoch')} {split} missing '{name}'")
                series[split][name].append(float(block[name]))
    return epochs, series


def draw_curves(epochs: list[int], series: dict[str, dict[str, list[float]]],
                output: Path, title: str | None = None) -> None:
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(8, 3.2 * len(METRICS)), sharex=True)
    if len(METRICS) == 1:
        axes = [axes]
    for ax, name in zip(axes, METRICS):
        ax.plot(epochs, series["train"][name], marker="o", label="train")
        ax.plot(epochs, series["validation"][name], marker="o", label="validation")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("epoch")
    fig.suptitle(title or "Train / validation metrics")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_zero_shot(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    """Index zero_shot.json rows by (dataset, horizon)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no results")
    indexed: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["dataset"]), int(row["horizon"]))
        indexed[key] = {name: float(row[name]) for name in ZS_METRICS}
    return indexed


def draw_zero_shot_compare(a: dict[tuple[str, int], dict[str, float]],
                           b: dict[tuple[str, int], dict[str, float]],
                           output: Path, label_a: str, label_b: str,
                           title: str | None = None) -> None:
    """4x2 grid: rows=horizons (96/192/336/720), cols=mse/mae; bars by dataset."""
    shared = set(a) & set(b)
    if not shared:
        raise ValueError("no shared (dataset, horizon) entries to compare")
    horizons = sorted({h for _, h in shared})
    datasets = sorted({ds for ds, _ in shared})
    width = 0.36
    x = np.arange(len(datasets))

    fig, axes = plt.subplots(
        len(horizons), len(ZS_METRICS),
        figsize=(4.2 * len(ZS_METRICS), 2.6 * len(horizons)),
        sharex="col", squeeze=False,
    )
    for i, horizon in enumerate(horizons):
        for j, metric in enumerate(ZS_METRICS):
            ax = axes[i, j]
            ya, yb = [], []
            for ds in datasets:
                key = (ds, horizon)
                if key not in shared:
                    ya.append(np.nan)
                    yb.append(np.nan)
                else:
                    ya.append(a[key][metric])
                    yb.append(b[key][metric])
            ax.bar(x - width / 2, ya, width, label=label_a)
            ax.bar(x + width / 2, yb, width, label=label_b)
            ax.set_title(f"H={horizon}  {metric}")
            ax.grid(True, axis="y", alpha=0.3)
            if i == 0 and j == len(ZS_METRICS) - 1:
                ax.legend(loc="upper right", fontsize=8)
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(horizons) - 1:
                ax.set_xticks(x, datasets, rotation=30, ha="right")
    fig.suptitle(title or "Zero-shot comparison")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/validation curves, or compare two zero-shot JSONs")
    parser.add_argument("--input-losses", type=Path, default=None,
                        help="losses.json written by sundial.cli train")
    parser.add_argument("--zero-shot-a", type=Path, default=None,
                        help="first zero_shot.json (results list)")
    parser.add_argument("--zero-shot-b", type=Path, default=None,
                        help="second zero_shot.json (results list)")
    parser.add_argument("--label-a", default=None, help="legend label for --zero-shot-a")
    parser.add_argument("--label-b", default=None, help="legend label for --zero-shot-b")
    parser.add_argument("--output", type=Path, default=None,
                        help="output image path (.png/.pdf/.svg)")
    args = parser.parse_args()

    zs_mode = args.zero_shot_a is not None or args.zero_shot_b is not None
    if zs_mode:
        if args.zero_shot_a is None or args.zero_shot_b is None:
            raise ValueError("both --zero-shot-a and --zero-shot-b are required")
        for path in (args.zero_shot_a, args.zero_shot_b):
            if not path.exists():
                raise FileNotFoundError(path)
        output = args.output or Path("outputs/zero_shot_compare.png")
        label_a = args.label_a or args.zero_shot_a.parent.name
        label_b = args.label_b or args.zero_shot_b.parent.name
        draw_zero_shot_compare(
            load_zero_shot(args.zero_shot_a), load_zero_shot(args.zero_shot_b),
            output, label_a, label_b,
            title=f"{label_a} vs {label_b}",
        )
        print(f"wrote {output}", flush=True)
        return

    # Default: train/validation loss curves
    input_losses = args.input_losses or Path("checkpoints/losses.json")
    output = args.output or Path("checkpoints/loss_curves.png")
    if not input_losses.exists():
        raise FileNotFoundError(input_losses)
    epochs, series = load_epoch_series(input_losses)
    draw_curves(epochs, series, output, title=str(input_losses))
    print(f"wrote {output} ({len(epochs)} epochs)", flush=True)


if __name__ == "__main__":
    main()
