"""Zero-shot ETT/Weather eval with the official HF sundial-base-128m.

Mirrors sundial.cli evaluate protocol (train-split z-score, same windows) so the
output JSON can be compared with our reproduction results.

Requires: pip install transformers==4.40.1

Run from the repository root:
    python src/original_model_test/zero_shot.py
    python src/original_model_test/zero_shot.py --data dataset/zero-shot --stride 20 --samples 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _split_borders(name: str, length: int, context: int) -> tuple[int, int, int]:
    # Same TSLib borders as sundial.cli.evaluate
    if name.startswith("ETTh"):
        train_end, val_end = 12 * 30 * 24, (12 + 4) * 30 * 24
    elif name.startswith("ETTm"):
        train_end, val_end = 12 * 30 * 24 * 4, (12 + 4) * 30 * 24 * 4
    else:
        train_end, val_end = int(length * 0.7), int(length * 0.8)
    if val_end >= length:
        raise ValueError(f"{name}: shorter than the standard TSLib train/validation split")
    return train_end, max(0, val_end - context), length


@torch.no_grad()
def _predict_batch(model, contexts: list[np.ndarray], horizon: int, samples: int,
                   device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.stack(contexts)).to(device)
    # revin=True: lookback instance-norm inside generate (matches our forecast)
    out = model.generate(x, max_new_tokens=horizon, num_samples=samples, revin=True)
    if out.ndim != 3:
        raise ValueError(f"unexpected generate shape {tuple(out.shape)}; expected [B,S,H]")
    return out.mean(1).cpu().numpy()


def evaluate(args: argparse.Namespace) -> None:
    _seed(args.seed)
    device = _device(args.device)
    print(f"loading {args.model} on {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=str(args.cache_dir),
    ).to(device)
    model.eval()

    results = []
    for name in DATASETS:
        path = args.data / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        cols = [c for c in frame.columns
                if c.lower() != "date" and pd.api.types.is_numeric_dtype(frame[c])]
        raw = frame[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{path} contains missing or infinite values")
        train_end, test_start, test_end = _split_borders(name, len(raw), args.context)
        mean, std = raw[:train_end].mean(0), raw[:train_end].std(0).clip(min=1e-6)
        series = (raw - mean) / std

        for horizon in args.horizons:
            if test_start + args.context + horizon > test_end:
                raise ValueError(
                    f"{name}: no test windows for context={args.context}, horizon={horizon}")
            sq, ab, count = 0.0, 0.0, 0
            contexts, targets = [], []

            def consume() -> None:
                nonlocal sq, ab, count
                if not contexts:
                    return
                pred = _predict_batch(model, contexts, horizon, args.samples, device)
                truth = np.stack(targets)
                error = pred - truth
                sq += float(np.square(error, dtype=np.float64).sum())
                ab += float(np.abs(error).sum())
                count += error.size
                contexts.clear()
                targets.clear()

            for start in range(test_start, test_end - args.context - horizon + 1, args.stride):
                past = series[start:start + args.context]
                future = series[start + args.context:start + args.context + horizon]
                for c in range(series.shape[1]):
                    contexts.append(past[:, c])
                    targets.append(future[:, c])
                    if len(contexts) >= args.batch_size:
                        consume()
            consume()

            row = {
                "dataset": name, "horizon": horizon,
                "mse": sq / count, "mae": ab / count,
                "windows_x_channels": count // horizon,
                "stride": args.stride, "samples": args.samples,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "results": results}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF sundial-base-128m zero-shot eval")
    parser.add_argument("--model", default="thuml/sundial-base-128m")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--data", type=Path, default=Path("dataset/zero-shot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/original_model/zero_shot.json"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--context", type=int, default=2880)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
