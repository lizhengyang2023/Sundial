"""Command-line entrypoints for corpus preparation, pretraining and zero-shot eval."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import BalancedCorpus, SOURCES, SundialIterableDataset, convert_source
from .model import Sundial, SundialConfig


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _load_config(path: Path | None) -> SundialConfig:
    if path is None:
        return SundialConfig()
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {x.name for x in fields(SundialConfig)}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"unknown Sundial config fields: {sorted(extra)}")
    return SundialConfig(**raw)


def _checkpoint(path: Path, model: Sundial, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                scaler: torch.amp.GradScaler, step: int, data_settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "data_settings": data_settings,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "step": step}, tmp)
    tmp.replace(path)


def _load_model(path: Path, device: torch.device) -> tuple[Sundial, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Sundial(SundialConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.to(device), state


def prepare(args: argparse.Namespace) -> None:
    manifest = convert_source(args.source, args.input, args.output,
                              shard_rows=args.shard_rows, read_batch_size=args.read_batch_size,
                              seed=args.seed,
                              train_fraction=args.train_fraction, clip_mad=args.clip_mad,
                              hf_cache=args.hf_cache)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def train(args: argparse.Namespace) -> None:
    if min(args.steps, args.steps_per_epoch, args.batch_size, args.accum) < 1 or args.workers < 0:
        raise ValueError("steps, steps_per_epoch, batch_size and accum must be positive; workers cannot be negative")
    _seed(args.seed)
    device = _device(args.device)
    if args.resume:
        model, state = _load_model(args.resume, device)
        cfg = model.cfg
        first_step = int(state["step"])
    else:
        cfg = _load_config(args.config)
        model = Sundial(cfg).to(device)
        state = None
        first_step = 0
    print(f"parameters={sum(p.numel() for p in model.parameters()):,} device={device}")
    data_settings = {"sources": args.sources, "batch_size": args.batch_size,
                     "accum": args.accum, "steps_per_epoch": args.steps_per_epoch,
                     "seed": args.seed}
    if state is not None and "data_settings" in state and state["data_settings"] != data_settings:
        raise ValueError("resume requires the same sources, batch size, accumulation, epoch length and seed")
    corpus = BalancedCorpus(args.corpus, cfg.patch_size, cfg.max_context, cfg.horizon,
                            seed=args.seed, sources=tuple(args.sources))
    dataset = SundialIterableDataset(corpus, args.batch_size,
                                     args.steps_per_epoch * args.accum, seed=args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, args.warmup)
    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, args.steps - warmup))
        return float(args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + np.cos(np.pi * progress)) / 2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    if state is not None:
        scaler.load_state_dict(state.get("scaler", {}))
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    model.train()
    step = first_step
    while step < args.steps:
        epoch = step // args.steps_per_epoch
        within_epoch = step % args.steps_per_epoch
        dataset.set_epoch(epoch, start_step=within_epoch * args.accum)
        loader_options = {"batch_size": None, "num_workers": args.workers,
                          "pin_memory": use_amp,
                          "generator": torch.Generator().manual_seed(args.seed + epoch)}
        if args.workers:
            loader_options["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(dataset, **loader_options)
        batches = iter(loader)
        until = min(args.steps, (epoch + 1) * args.steps_per_epoch)
        while step < until:
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for _ in range(args.accum):
                batch = {k: v.to(device, non_blocking=use_amp) for k, v in next(batches).items()}
                autocast = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()
                with autocast:
                    loss = model.training_loss(**batch) / args.accum
                scaler.scale(loss).backward()
                running += loss.detach().float().item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % args.log_every == 0 or step == first_step + 1:
                print(f"step={step} loss={running:.6f} lr={scheduler.get_last_lr()[0]:.3g}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                _checkpoint(args.checkpoint, model, optimizer, scheduler, scaler, step, data_settings)
        del batches, loader


def _split_borders(name: str, length: int, context: int) -> tuple[int, int, int]:
    if name.startswith("ETTh"):
        train_end, val_end = 12 * 30 * 24, (12 + 4) * 30 * 24
    elif name.startswith("ETTm"):
        train_end, val_end = 12 * 30 * 24 * 4, (12 + 4) * 30 * 24 * 4
    else:
        train_end, val_end = int(length * 0.7), int(length * 0.8)
    if val_end >= length:
        raise ValueError(f"{name}: shorter than the standard TSLib train/validation split")
    return train_end, max(0, val_end - context), length


def evaluate(args: argparse.Namespace) -> None:
    _seed(args.seed)
    device = _device(args.device)
    model, state = _load_model(args.checkpoint, device)
    model.eval()
    cfg = model.cfg
    results = []
    for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"):
        path = args.data / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        cols = [c for c in frame.columns if c.lower() != "date" and pd.api.types.is_numeric_dtype(frame[c])]
        raw = frame[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"{path} contains missing or infinite values; clean evaluation data first")
        train_end, test_start, test_end = _split_borders(name, len(raw), args.context)
        mean, std = raw[:train_end].mean(0), raw[:train_end].std(0).clip(min=1e-6)
        series = (raw - mean) / std
        for horizon in args.horizons:
            if test_start + args.context + horizon > test_end:
                raise ValueError(f"{name}: no test windows for context={args.context}, horizon={horizon}")
            sq, ab, count = 0.0, 0.0, 0
            contexts, targets = [], []
            def consume() -> None:
                nonlocal sq, ab, count
                if not contexts:
                    return
                x = torch.from_numpy(np.stack(contexts)).to(device)
                pred = model.forecast(x, horizon, args.samples, steps=args.flow_steps).mean(1).cpu().numpy()
                truth = np.stack(targets)
                error = pred - truth
                sq += float(np.square(error, dtype=np.float64).sum())
                ab += float(np.abs(error).sum())
                count += error.size
                contexts.clear(); targets.clear()
            with torch.no_grad():
                for start in range(test_start, test_end - args.context - horizon + 1, args.stride):
                    past = series[start:start + args.context]
                    future = series[start + args.context:start + args.context + horizon]
                    for c in range(series.shape[1]):
                        contexts.append(past[:, c])
                        targets.append(future[:, c])
                        if len(contexts) >= args.batch_size:
                            consume()
                consume()
            row = {"dataset": name, "horizon": horizon, "mse": sq / count,
                   "mae": ab / count, "windows_x_channels": count // horizon,
                   "stride": args.stride, "samples": args.samples}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint_step": state["step"], "results": results},
                                      indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sundial.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="convert one source to S3 Parquet")
    prep.add_argument("--source", choices=SOURCES, required=True)
    prep.add_argument("--input", required=True,
                      help="local directory/file or hf://datasets/owner/repo/subdirectory")
    prep.add_argument("--output", type=Path, default=Path("corpus"))
    prep.add_argument("--hf-cache", type=Path, default=Path(".cache/huggingface"))
    prep.add_argument("--shard-rows", type=int, default=128)
    prep.add_argument("--read-batch-size", type=int, default=64)
    prep.add_argument("--train-fraction", type=float, default=0.9)
    prep.add_argument("--clip-mad", type=float, default=10.0)
    prep.add_argument("--seed", type=int, default=42)
    prep.set_defaults(func=prepare)
    fit = sub.add_parser("train", help="train paper-level Sundial on balanced S3 corpus")
    fit.add_argument("--corpus", type=Path, default=Path("corpus"))
    fit.add_argument("--config", type=Path, help="JSON override; default is Sundial Base F=720")
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sundial-base.pt"))
    fit.add_argument("--steps", type=int, required=True)
    fit.add_argument("--steps-per-epoch", type=int, default=1000)
    fit.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    fit.add_argument("--batch-size", type=int, default=6)
    fit.add_argument("--accum", type=int, default=1)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--prefetch-factor", type=int, default=2)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=0.1)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--min-lr-ratio", type=float, default=0.1)
    fit.add_argument("--clip-grad", type=float, default=1.0)
    fit.add_argument("--log-every", type=int, default=10)
    fit.add_argument("--save-every", type=int, default=1000)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default="auto")
    fit.set_defaults(func=train)
    ev = sub.add_parser("evaluate", help="ETT/Weather zero-shot point forecasting")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--data", type=Path, default=Path("dataset"))
    ev.add_argument("--output", type=Path, default=Path("results/zero_shot.json"))
    ev.add_argument("--horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    ev.add_argument("--context", type=int, default=2880)
    ev.add_argument("--samples", type=int, default=1)
    ev.add_argument("--flow-steps", type=int, default=50)
    ev.add_argument("--stride", type=int, default=1)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
