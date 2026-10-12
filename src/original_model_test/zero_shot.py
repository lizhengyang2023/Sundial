"""Zero-shot ETT/Weather eval for official Sundial (HF safetensors or GGUF).

Mirrors sundial.cli evaluate protocol (train-split z-score, same windows) so the
output JSON can be compared with our reproduction results.

Backends:
  hf   — transformers AutoModelForCausalLM (thuml/sundial-base-128m)
         Requires: pip install transformers==4.40.1
  gguf — zsfm.SundialModel (amaye15/sundial-gguf); deterministic single path
         Requires: pip install zsfm

Run from the repository root:
    python src/original_model_test/zero_shot.py
    python src/original_model_test/zero_shot.py --backend gguf
    python src/original_model_test/zero_shot.py --backend gguf --gguf /data/amaye15/sundial-gguf/sundial-f16.gguf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")
DEFAULT_GGUF = Path("/data/amaye15/sundial-gguf")


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


def _resolve_gguf(path: Path) -> Path:
    """Accept a .gguf file or a directory containing sundial-*.gguf."""
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"expected a .gguf file, got {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    preferred = ("sundial-f16.gguf", "sundial-f32.gguf", "sundial-q8.gguf")
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            return candidate
    found = sorted(path.glob("*.gguf"))
    if not found:
        raise FileNotFoundError(f"no .gguf under {path}")
    return found[0]


def _load_hf(model_id: str, cache_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=str(cache_dir),
    ).to(device)
    model.eval()
    return model


def _load_gguf(path: Path):
    import zsfm
    gguf = _resolve_gguf(path)
    print(f"resolved gguf={gguf}", flush=True)
    return zsfm.SundialModel(str(gguf)), gguf


@torch.no_grad()
def _predict_hf(model, contexts: list[np.ndarray], horizon: int, samples: int,
                device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.stack(contexts)).to(device)
    # revin=True: lookback instance-norm inside generate (matches our forecast)
    out = model.generate(x, max_new_tokens=horizon, num_samples=samples, revin=True)
    if out.ndim != 3:
        raise ValueError(f"unexpected generate shape {tuple(out.shape)}; expected [B,S,H]")
    return out.mean(1).cpu().numpy()


def _predict_gguf(model, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
    # zsfm is univariate + deterministic (zero-init ODE); one series per call
    preds = []
    for ctx in contexts:
        point = model.forecast(ctx.astype(np.float32).tolist(), horizon=horizon)
        arr = np.asarray(point, dtype=np.float32).reshape(-1)
        if arr.shape[0] < horizon:
            raise ValueError(f"gguf forecast shorter than horizon: {arr.shape[0]} < {horizon}")
        preds.append(arr[:horizon])
    return np.stack(preds)


def evaluate(args: argparse.Namespace) -> None:
    _seed(args.seed)
    device = _device(args.device)

    if args.backend == "hf":
        print(f"loading hf model={args.model} on {device}", flush=True)
        model = _load_hf(args.model, args.cache_dir, device)
        model_tag = args.model
        samples_used = args.samples
    else:
        print(f"loading gguf from {args.gguf}", flush=True)
        model, gguf_path = _load_gguf(args.gguf)
        model_tag = str(gguf_path)
        samples_used = 1  # GGUF port has no multi-sample ensemble
        if args.samples != 1:
            print(f"note: gguf ignores --samples={args.samples}; using deterministic path",
                  flush=True)

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
                if args.backend == "hf":
                    pred = _predict_hf(model, contexts, horizon, samples_used, device)
                else:
                    pred = _predict_gguf(model, contexts, horizon)
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
                "stride": args.stride, "samples": samples_used,
                "backend": args.backend,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"backend": args.backend, "model": model_tag, "results": results}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official Sundial zero-shot eval (HF or GGUF)")
    parser.add_argument("--backend", choices=("hf", "gguf"), default="hf",
                        help="hf: transformers safetensors; gguf: zsfm local file")
    parser.add_argument("--model", default="thuml/sundial-base-128m",
                        help="HF model id or local snapshot dir (backend=hf)")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF,
                        help="GGUF file or directory (backend=gguf)")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--data", type=Path, default=Path("dataset/zero-shot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/original_model/zero_shot.json"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--context", type=int, default=2880)
    parser.add_argument("--samples", type=int, default=20,
                        help="HF multi-sample mean; ignored for gguf (deterministic)")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="used by backend=hf only")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
