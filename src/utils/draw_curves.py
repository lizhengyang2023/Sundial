"""Plot per-epoch train/validation metrics from a losses.json file.

Run from the repository root:
    python src/utils/draw_curves.py
    python src/utils/draw_curves.py --input-losses checkpoints/losses.json --output checkpoints/loss_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("flow_loss", "mae", "mse")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation curves from losses.json")
    parser.add_argument("--input-losses", type=Path, default=Path("checkpoints/losses.json"),
                        help="losses.json written by sundial.cli train")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/loss_curves.png"),
                        help="output image path (.png/.pdf/.svg)")
    args = parser.parse_args()
    if not args.input_losses.exists():
        raise FileNotFoundError(args.input_losses)
    epochs, series = load_epoch_series(args.input_losses)
    draw_curves(epochs, series, args.output, title=str(args.input_losses))
    print(f"wrote {args.output} ({len(epochs)} epochs)", flush=True)


if __name__ == "__main__":
    main()
