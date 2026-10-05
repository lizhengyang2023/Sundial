"""Discover local/Hub data files and convert time series to S3 Parquet."""

from __future__ import annotations

# --- before any huggingface_hub / httpx / requests importing ---
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    import warnings
    warnings.warn("truststore 未安装，若在内网/代理环境可能继续报 SSL 错误。"
                  "请运行：pip install truststore")

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import json
import logging
import os
import random
import re
import time
from typing import Callable, Iterator

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
logger = logging.getLogger(__name__)


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


def discover_hf_files(root: str, cache_dir: Path,
                      progress: Progress | None = None) -> Iterator[tuple[str, Callable[[], Path]]]:
    """List Hub files by page; defer each download until the file is processed.

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
    revision = progress.revision if progress and progress.revision else api.repo_info(repo_id, repo_type="dataset").sha
    if progress and not progress.revision:
        progress.revision = revision
        progress.save()
    for entry in api.list_repo_tree(repo_id, path_in_repo=subdirectory,
                                    recursive=True, revision=revision, repo_type="dataset"):
        if not isinstance(entry, RepoFile) or Path(entry.path).suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if _is_eval_name(entry.path):
            continue
        filename = entry.path
        def download(filename: str = filename) -> Path:
            return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                        repo_type="dataset", revision=revision,
                                        cache_dir=cache_dir))
        yield filename, download


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


class Progress:
    """Durable state for completed files and shards of one conversion."""

    def __init__(self, dest: Path, settings: dict):
        self.dest = dest
        self.path = dest / "progress.json"
        self.settings = settings
        self.done: set[str] = set()
        self.failed: dict[str, str] = {}
        self.attempts: dict[str, int] = {}
        self.shards: dict[str, list[dict]] = {"train": [], "validation": []}
        self.skipped = {"evaluation": 0, "short_or_empty": 0}
        self.active: dict | None = None
        self.revision: str | None = None
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("settings") != settings:
                raise ValueError("prepare settings differ from progress.json; use the original settings or a new output directory")
            self.done = set(data["done"])
            self.failed = dict(data["failed"])
            self.attempts = dict(data["attempts"])
            self.shards = data["shards"]
            self.skipped = data["skipped"]
            self.active = data.get("active")
            self.revision = data.get("revision")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "settings": self.settings, "revision": self.revision,
            "done": sorted(self.done), "failed": self.failed,
            "attempts": self.attempts, "shards": self.shards,
            "skipped": self.skipped, "active": self.active,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def begin(self, relative: str) -> None:
        self.active = {"file": relative,
                       "counts": {split: len(items) for split, items in self.shards.items()}}
        self.save()

    def record_shard(self, split: str, path: str, rows: int) -> None:
        self.shards[split].append({"path": path, "rows": rows})
        self.save()

    def mark_done(self, relative: str, skipped: dict[str, int]) -> None:
        old_skipped = self.skipped
        old_active = self.active
        old_failure = self.failed.pop(relative, None)
        old_attempts = self.attempts.pop(relative, None)
        self.done.add(relative)
        self.skipped = {key: self.skipped[key] + skipped[key] for key in self.skipped}
        self.active = None
        try:
            self.save()
        except Exception:
            self.done.discard(relative)
            self.skipped = old_skipped
            self.active = old_active
            if old_failure is not None:
                self.failed[relative] = old_failure
            if old_attempts is not None:
                self.attempts[relative] = old_attempts
            raise

    def mark_failed(self, relative: str, exc: Exception) -> None:
        self.failed[relative] = f"{type(exc).__name__}: {exc}"
        self.active = None
        self.save()

    def rollback(self) -> None:
        """Remove every shard after the last completed-file boundary."""
        if self.active is None:
            return
        for split, count in self.active["counts"].items():
            self.shards[split] = self.shards[split][:count]
        self.active = None
        self.save()
        self.cleanup_orphans()

    def cleanup_orphans(self) -> None:
        """Remove files written after a crash but absent from progress.json."""
        for split, items in self.shards.items():
            folder = self.dest / split
            expected = {Path(item["path"]).name for item in items}
            missing = [name for name in expected if not (folder / name).is_file()]
            if missing:
                raise FileNotFoundError(f"progress.json references missing {split} shard: {missing[0]}")
            if not folder.exists():
                continue
            for path in folder.glob("part-*.parquet"):
                if path.name not in expected:
                    path.unlink()
            for path in folder.glob("part-*.tmp"):
                path.unlink()


class ShardWriter:
    """Shuffle a bounded number of series and write one Parquet shard at a time."""

    def __init__(self, dest: Path, shard_rows: int = 128, seed: int = 42,
                 progress: Progress | None = None):
        if shard_rows < 1:
            raise ValueError("shard_rows must be positive")
        self.dest = dest
        self.shard_rows = shard_rows
        self.seed = seed
        self.progress = progress
        self.buffers: dict[str, list[dict]] = {"train": [], "validation": []}
        self.shards: dict[str, list[dict]] = (progress.shards if progress is not None
                                              else {"train": [], "validation": []})

    def add(self, split: str, series_id: str, values: np.ndarray) -> None:
        self.buffers[split].append({"series_id": series_id, "values": values.tolist()})
        if len(self.buffers[split]) >= self.shard_rows:
            self._flush(split)

    def _flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        random.Random(f"{self.seed}:{split}:{len(self.shards[split])}").shuffle(rows)
        folder = self.dest / split
        folder.mkdir(parents=True, exist_ok=True)
        name = f"part-{len(self.shards[split]):06d}.parquet"
        path = folder / name
        temporary = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        temporary.replace(path)
        if self.progress is not None:
            self.progress.record_shard(split, f"{split}/{name}", len(rows))
            self.shards = self.progress.shards
        else:
            self.shards[split].append({"path": f"{split}/{name}", "rows": len(rows)})
        rows.clear()

    def flush(self) -> None:
        for split in self.buffers:
            self._flush(split)

    def finalize(self) -> dict:
        self.flush()
        return {"shards": self.shards,
                "train_series": sum(s["rows"] for s in self.shards["train"]),
                "validation_series": sum(s["rows"] for s in self.shards["validation"])}


def convert_source(source: str, input_path: Path | str, output_root: Path,
                   *, shard_rows: int = 128, read_batch_size: int = 64,
                   seed: int = 42,
                   train_fraction: float = 0.9, clip_mad: float = 10.0,
                   hf_cache: Path = Path(".cache/huggingface"),
                   resume: bool = True, max_file_retries: int = 3) -> dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if (not 0 < train_fraction < 1 or shard_rows < 1 or read_batch_size < 1
            or clip_mad <= 0 or max_file_retries < 1):
        raise ValueError("invalid preprocessing parameters")
    remote = str(input_path).startswith("hf://")
    if not remote:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.is_dir() and output_root.resolve().is_relative_to(input_path.resolve()):
            raise ValueError("output directory must be outside the input directory")
    dest = output_root / source
    if (dest / "manifest.json").exists():
        raise FileExistsError(f"{dest} already has a manifest; use a new output directory")
    if (dest / "progress.json").exists() and not resume:
        raise FileExistsError(f"{dest} has progress; use a new output directory for a fresh conversion")
    if not (dest / "progress.json").exists() and dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} has data without progress.json; use a new output directory")
    settings = {"source": source, "input": str(input_path), "shard_rows": shard_rows,
                "read_batch_size": read_batch_size, "seed": seed,
                "train_fraction": train_fraction, "clip_mad": clip_mad}
    progress = Progress(dest, settings)
    progress.rollback()
    progress.cleanup_orphans()
    writer = ShardWriter(dest, shard_rows, seed, progress)
    if remote:
        files = discover_hf_files(str(input_path), hf_cache, progress)
    else:
        files = ((str(path.relative_to(input_path)) if Path(input_path).is_dir() else path.name,
                  lambda path=path: path) for path in discover_files(Path(input_path)))

    seen_file = False
    seen_names: set[str] = set()
    for relative, get_path in files:
        seen_file = True
        seen_names.add(relative)
        if relative in progress.done:
            continue
        if _is_eval_name(relative):
            progress.mark_done(relative, {"evaluation": 1, "short_or_empty": 0})
            continue
        for attempt in range(max_file_retries):
            progress.begin(relative)
            file_skipped = {"evaluation": 0, "short_or_empty": 0}
            try:
                path = get_path()
                for row_number, row in enumerate(iter_rows(path, read_batch_size)):
                    series_id = str(row.get("id", row.get("item_id", row_number)))
                    if _is_eval_name(series_id):
                        file_skipped["evaluation"] += 1
                        continue
                    for variate, values in extract_univariate(row):
                        prepared = normalize_per_series(values, train_fraction, clip_mad)
                        if prepared is None:
                            file_skipped["short_or_empty"] += 1
                            continue
                        for split, series in zip(("train", "validation"), prepared):
                            if len(series) >= 16:
                                writer.add(split, f"{relative}:{series_id}:{row_number}:{variate}", series)
                writer.flush()
                progress.mark_done(relative, file_skipped)
                break
            except Exception as exc:
                writer.buffers = {"train": [], "validation": []}
                progress.rollback()
                progress.attempts[relative] = progress.attempts.get(relative, 0) + 1
                progress.mark_failed(relative, exc)
                if attempt + 1 < max_file_retries:
                    delay = 2 ** attempt
                    logger.warning("File %s failed (attempt %d/%d): %s; retrying in %ds",
                                   relative, attempt + 1, max_file_retries, exc, delay)
                    time.sleep(delay)
                else:
                    logger.error("File %s failed after %d attempts: %s",
                                 relative, max_file_retries, exc)
    if progress.done - seen_names:
        raise ValueError(f"previously completed files are missing from input: {sorted(progress.done - seen_names)[:3]}")
    if progress.failed:
        raise RuntimeError(f"{len(progress.failed)} file(s) failed; rerun prepare to retry them; details in {progress.path}")
    manifest = writer.finalize()
    if not seen_file:
        raise ValueError(f"no {sorted(SUPPORTED_SUFFIXES)} files under {input_path}")
    if not manifest["train_series"]:
        raise ValueError("no training series survived conversion")
    manifest.update({"source": source, "seed": seed, "train_fraction": train_fraction,
                     "clip_mad": clip_mad, "skipped": progress.skipped})
    temporary = dest / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(dest / "manifest.json")
    progress.path.unlink()
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
