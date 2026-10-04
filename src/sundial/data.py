"""Discover local/Hub data files and convert time series to S3 Parquet."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import json
import os
import random
import re
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info


SOURCES = ("utsd", "chronos", "lotsa")
SUPPORTED_SUFFIXES = {".parquet", ".arrow"}
EVAL_NAMES = re.compile(
    r"(^|[/\\_.-])(etth1|etth2|ettm1|ettm2|etth|ettm|weather|electricity|traffic|illness|exchange)([/\\_.-]|$)",
    re.I
)
META_COLUMNS = {"id", "item_id", "timestamp", "time", "datetime", "time_idx",
                "ds", "start", "end", "freq", "date", "category",
                "series_name", "target_name"}


def _is_eval_name(name: str) -> bool:
    return bool(EVAL_NAMES.search(name))


def discover_files(root: Path) -> Iterator[Path]:
    """Yield supported local files without building a complete file list."""
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        path = Path(entry.path)
                        if path.suffix.lower() in SUPPORTED_SUFFIXES:
                            yield path
        except (PermissionError, FileNotFoundError):
            continue


def discover_hf_files(root: str, cache_dir: Path) -> Iterator[tuple[str, Path]]:
    """List Hub files by page; fetch each matching file only when consumed.

    root format: hf://datasets/<owner>/<repo>[/<subdirectory>].
    The path prefix is important for UTSD: select UTSD-1G rather than all tiers.
    """
    prefix = "hf://datasets/"
    if not root.startswith(prefix):
        raise ValueError(f"Hugging Face input must start with {prefix}")
    parts = root[len(prefix):].strip("/").split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError("Hugging Face input needs an owner and dataset repo")
    repo_id = "/".join(parts[:2])
    subdirectory = "/".join(parts[2:]) or None
    # Xet has a separate cache. Keep it alongside the requested Hub cache.
    os.environ.setdefault("HF_XET_CACHE", str((cache_dir / "xet").resolve()))
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    revision = api.repo_info(repo_id, repo_type="dataset").sha
    for entry in api.list_repo_tree(repo_id, path_in_repo=subdirectory,
                                    recursive=True, revision=revision, repo_type="dataset"):
        if not isinstance(entry, RepoFile) or Path(entry.path).suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if _is_eval_name(entry.path):
            continue
        local = hf_hub_download(repo_id=repo_id, filename=entry.path,
                                repo_type="dataset", revision=revision,
                                cache_dir=cache_dir)
        yield entry.path, Path(local)


def extract_univariate(row: dict) -> Iterator[tuple[str, np.ndarray]]:
    """Extract 1-D series from numeric vector or matrix columns, skipping metadata."""
    for key, value in row.items():
        if key.lower() in META_COLUMNS or value is None:
            continue
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            yield key, arr
        elif arr.ndim == 2:
            # The shorter axis is assumed to be the number of variables.
            channel_first = arr.shape[0] <= arr.shape[1]
            for i, series in enumerate(arr if channel_first else arr.T):
                yield f"{key}:{i}", np.asarray(series, dtype=np.float32)


def iter_rows(path: Path, read_batch_size: int = 64) -> Iterator[dict]:
    """Read one Arrow/Parquet file through HF Datasets without materializing it."""
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be positive")
    kind = path.suffix.lower().lstrip(".")
    if kind not in {"arrow", "parquet"}:
        raise ValueError(f"unsupported data file: {path}")
    dataset = load_dataset(kind, data_files={"train": str(path)},
                           split="train", streaming=True)
    for batch in dataset.batch(batch_size=read_batch_size):
        columns = tuple(batch)
        for i in range(len(batch[columns[0]]) if columns else 0):
            yield {name: batch[name][i] for name in columns}


def _clean_split(values: np.ndarray, lower: float, upper: float, fill: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan
    # Missing points use the last observed value; leading gaps use train median.
    x = pd.Series(x).ffill().fillna(fill).to_numpy()
    return np.clip(x, lower, upper).astype(np.float32)


def normalize_per_series(values: np.ndarray, train_fraction: float,
                         clip_mad: float) -> tuple[np.ndarray, np.ndarray] | None:
    if values.ndim != 1 or len(values) < 32:
        return None
    cut = int(len(values) * train_fraction)
    train_raw = np.asarray(values[:cut], dtype=np.float64)
    finite = train_raw[np.isfinite(train_raw)]
    if len(finite) < 32:
        return None
    median = float(np.median(finite))
    robust_std = max(1.4826 * float(np.median(np.abs(finite - median))), 1e-6)
    lower, upper = median - clip_mad * robust_std, median + clip_mad * robust_std
    train = _clean_split(values[:cut], lower, upper, median)
    valid = _clean_split(values[cut:], lower, upper, median)
    mean, std = float(train.mean()), max(float(train.std()), 1e-6)
    return ((train - mean) / std).astype(np.float32), ((valid - mean) / std).astype(np.float32)


class ShardWriter:
    """Shuffle a bounded number of series and write one Parquet shard at a time."""

    def __init__(self, dest: Path, shard_rows: int = 128, seed: int = 42):
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.dest = dest
        self.shard_rows = shard_rows
        self.rng = random.Random(seed)
        self.buffers: dict[str, list[dict]] = {"train": [], "validation": []}
        self.shards: dict[str, list[dict]] = {"train": [], "validation": []}

    def add(self, split: str, series_id: str, values: np.ndarray) -> None:
        self.buffers[split].append({"series_id": series_id, "values": values.tolist()})
        if len(self.buffers[split]) >= self.shard_rows:
            self._flush(split)

    def _flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        self.rng.shuffle(rows)
        folder = self.dest / split
        folder.mkdir(parents=True, exist_ok=True)
        name = f"part-{len(self.shards[split]):06d}.parquet"
        path = folder / name
        temporary = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        temporary.replace(path)
        self.shards[split].append({"path": f"{split}/{name}", "rows": len(rows)})
        rows.clear()

    def finalize(self) -> dict:
        for split in self.buffers:
            self._flush(split)
        return {"shards": self.shards,
                "train_series": sum(s["rows"] for s in self.shards["train"]),
                "validation_series": sum(s["rows"] for s in self.shards["validation"])}


def convert_source(source: str, input_path: Path | str, output_root: Path,
                   *, shard_rows: int = 128, read_batch_size: int = 64,
                   seed: int = 42,
                   train_fraction: float = 0.9, clip_mad: float = 10.0,
                   hf_cache: Path = Path(".cache/huggingface")) -> dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if not 0 < train_fraction < 1 or shard_rows < 1 or read_batch_size < 1 or clip_mad <= 0:
        raise ValueError("invalid preprocessing parameters")
    remote = str(input_path).startswith("hf://")
    if remote:
        files = discover_hf_files(str(input_path), hf_cache)
    else:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.is_dir() and output_root.resolve().is_relative_to(input_path.resolve()):
            raise ValueError("output directory must be outside the input directory")
        files = ((str(path.relative_to(input_path)) if input_path.is_dir() else path.name, path)
                 for path in discover_files(input_path))
    dest = output_root / source
    if (dest / "manifest.json").exists():
        raise FileExistsError(f"{dest} already has a manifest; use a new output directory")
    writer = ShardWriter(dest, shard_rows, seed)
    skipped = {"evaluation": 0, "short_or_empty": 0}

    seen_file = False
    for relative, path in files:
        seen_file = True
        if _is_eval_name(relative):
            skipped["evaluation"] += 1
            continue
        for row_number, row in enumerate(iter_rows(path, read_batch_size)):
            series_id = str(row.get("id", row.get("item_id", row_number)))
            if _is_eval_name(series_id):
                skipped["evaluation"] += 1
                continue
            for variate, values in extract_univariate(row):
                prepared = normalize_per_series(values, train_fraction, clip_mad)
                if prepared is None:
                    skipped["short_or_empty"] += 1
                    continue
                for split, series in zip(("train", "validation"), prepared):
                    if len(series) < 16:
                        continue
                    writer.add(split, f"{relative}:{series_id}:{row_number}:{variate}", series)
    manifest = writer.finalize()
    if not seen_file:
        raise ValueError(f"no {sorted(SUPPORTED_SUFFIXES)} files under {input_path}")
    if not manifest["train_series"]:
        raise ValueError("no training series survived conversion")
    manifest.update({"source": source, "seed": seed, "train_fraction": train_fraction,
                     "clip_mad": clip_mad, "skipped": skipped})
    temporary = dest / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(dest / "manifest.json")
    return manifest


class BalancedCorpus:
    """Equal source counts, uniform series choice and random rolling windows."""

    def __init__(self, root: Path, patch_size: int, max_context: int, horizon: int,
                 seed: int = 42, split: str = "train", cache_shards: int = 8,
                 sources: tuple[str, ...] = SOURCES):
        if not sources or len(set(sources)) != len(sources) or any(s not in SOURCES for s in sources):
            raise ValueError(f"sources must be a nonempty subset of {SOURCES}")
        if cache_shards < 1 or patch_size < 1 or max_context < patch_size or horizon < 1:
            raise ValueError("invalid sampling parameters")
        self.root, self.patch_size = root, patch_size
        self.max_context, self.horizon = max_context, horizon
        self.source_names = sources
        self.rng = random.Random(seed)
        self.sources: dict[str, tuple[list[dict], list[int], int]] = {}
        self.cache: OrderedDict[str, list[dict]] = OrderedDict()
        self.cache_shards = cache_shards
        for source in sources:
            path = root / source / "manifest.json"
            if not path.exists():
                raise FileNotFoundError(f"missing {source} manifest: {path}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            shards = manifest["shards"][split]
            cumulative = []
            total = 0
            for shard in shards:
                total += shard["rows"]
                cumulative.append(total)
            if not total:
                raise ValueError(f"no {split} series for {source}")
            self.sources[source] = shards, cumulative, total

    def _choose_series(self, source: str) -> np.ndarray:
        shards, cumulative, total = self.sources[source]
        idx = self.rng.randrange(total)
        shard_idx = bisect_right(cumulative, idx)
        offset = cumulative[shard_idx - 1] if shard_idx else 0
        path = str(self.root / source / shards[shard_idx]["path"])
        if path not in self.cache:
            self.cache[path] = pq.read_table(path).to_pylist()
            if len(self.cache) > self.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return np.asarray(self.cache[path][idx - offset]["values"], dtype=np.float32)

    def _cut_window(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        full_future = len(series) >= self.patch_size + self.horizon
        reserve = self.horizon if full_future else 1
        available = min(len(series) - reserve, self.max_context)
        max_patches = available // self.patch_size
        if max_patches < 1:
            return None
        context_len = self.rng.randint(1, max_patches) * self.patch_size
        start = self.rng.randint(0, len(series) - context_len - reserve)
        stop = start + context_len
        return series[start:stop], series[stop:stop + self.horizon]

    def sample(self, batch_size: int, step: int) -> dict[str, torch.Tensor]:
        if batch_size < len(self.source_names) or batch_size % len(self.source_names):
            raise ValueError(f"batch_size must be a positive multiple of {len(self.source_names)}")
        per_source = batch_size // len(self.source_names)
        items = []
        for source in self.source_names:
            collected = 0
            for _ in range(per_source * 10):
                if collected == per_source:
                    break
                window = self._cut_window(self._choose_series(source))
                if window is not None:
                    items.append(window)
                    collected += 1
            if collected < per_source:
                raise RuntimeError(f"failed to sample {per_source} windows from {source}")
        length = max(len(x[0]) for x in items)
        f = self.horizon
        context = np.zeros((len(items), length), dtype=np.float32)
        context_valid = np.zeros_like(context, dtype=bool)
        full = np.zeros((len(items), length + f), dtype=np.float32)
        full_valid = np.zeros_like(full, dtype=bool)
        for j, (history, future) in enumerate(items):
            offset = length - len(history)
            context[j, offset:] = history
            context_valid[j, offset:] = True
            full[j, offset:length] = history
            full_valid[j, offset:length] = True
            full[j, length:length + len(future)] = future
            full_valid[j, length:length + len(future)] = True
        permutation = list(range(len(items)))
        self.rng.shuffle(permutation)
        return {"context": torch.from_numpy(context[permutation]),
                "context_valid": torch.from_numpy(context_valid[permutation]),
                "full_values": torch.from_numpy(full[permutation]),
                "full_valid": torch.from_numpy(full_valid[permutation])}


class SundialIterableDataset(IterableDataset):
    """Yield complete batches; step seeds avoid duplicates across DataLoader workers."""

    def __init__(self, corpus: BalancedCorpus, batch_size: int,
                 steps_per_epoch: int, seed: int = 42):
        super().__init__()
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        self.corpus = corpus
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        self.start_step = 0

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        if epoch < 0 or not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError("invalid epoch or start_step")
        self.epoch = epoch
        self.start_step = start_step

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        for step in range(self.start_step + worker_id, self.steps_per_epoch, worker_count):
            self.corpus.rng.seed(self.seed + self.epoch * self.steps_per_epoch + step)
            yield self.corpus.sample(self.batch_size, step)
