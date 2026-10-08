"""Inventory Chronos/LOTSA without downloading their data files.

Run from the repository root: python src/utils/data_check.py
MB means 1,000,000 bytes. Estimates are for choosing directories, not exact
counts of series that will survive Sundial's preprocessing.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, HfFileSystem, hf_hub_url
from huggingface_hub.hf_api import RepoFile


REPOS = {
    "chronos": "autogluon/chronos_datasets",
    "lotsa": "Salesforce/lotsa_data",
}
EVAL_NAMES = re.compile(
    r"(^|[/\\_.-])(etth1|etth2|ettm1|ettm2|etth|ettm|weather|electricity|traffic|illness|exchange)([/\\_.-]|$)",
    re.I,
)
VALUE_BYTES = {"float16": 2, "float32": 4, "float64": 8,
               "int16": 2, "int32": 4, "int64": 8}
META_COLUMNS = {"id", "item_id", "timestamp", "time", "datetime", "time_idx",
                "ds", "start", "end", "freq", "date", "category",
                "series_name", "target_name"}


def mb(size: int | float) -> float:
    return round(size / 1_000_000, 3)


def parquet_target_type(field_type: pa.DataType) -> tuple[int, str]:
    depth = 0
    while (pa.types.is_list(field_type) or pa.types.is_large_list(field_type)
           or pa.types.is_fixed_size_list(field_type)):
        depth += 1
        field_type = field_type.value_type
    return depth, str(field_type)


def first_parquet_estimate(repo: str, revision: str, files: list[RepoFile]) -> dict:
    first = min(files, key=lambda item: item.path)
    remote_path = f"datasets/{repo}@{revision}/{first.path}"
    with HfFileSystem().open(remote_path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
        schema = parquet.schema_arrow
        value_fields = []
        sample_channels = 0
        for field in schema:
            depth, dtype = parquet_target_type(field.type)
            if (field.name.lower() in META_COLUMNS or depth not in (1, 2)
                    or not (pa.types.is_integer(field.type.value_type if depth == 1 else
                                                field.type.value_type.value_type)
                            or pa.types.is_floating(field.type.value_type if depth == 1 else
                                                    field.type.value_type.value_type))):
                continue
            channels_per_row = 1
            if depth == 2:
                first_batch = next(parquet.iter_batches(batch_size=1, columns=[field.name]))
                value = first_batch.column(0)[0].as_py()
                if not value or not value[0]:
                    raise ValueError(f"cannot infer channels from empty 2-D field {field.name}")
                channels_per_row = min(len(value), len(value[0]))
            sample_channels += metadata.num_rows * channels_per_row
            value_fields.append((field.name, dtype, channels_per_row))
        if not value_fields:
            raise ValueError("first Parquet has no numeric sequence columns")
        value_names = {field[0] for field in value_fields}
        value_columns = [index for index in range(metadata.row_group(0).num_columns)
                         if metadata.row_group(0).column(index).path_in_schema.split(".")[0] in value_names]
        sample_points = sum(metadata.row_group(group).column(index).num_values
                            for group in range(metadata.num_row_groups)
                            for index in value_columns)
        sample_rows = metadata.num_rows
    if not sample_rows or not first.size:
        raise ValueError("first Parquet has no rows or no recorded size")
    total_bytes = sum(item.size for item in files)
    ratio = total_bytes / first.size
    return {
        "method": "first_parquet_metadata_scaled_by_parquet_bytes",
        "sample_file": first.path,
        "sample_file_mb": mb(first.size),
        "sample_rows": sample_rows,
        "sample_channels": sample_channels,
        "sample_numeric_values": sample_points,
        "numeric_sequence_fields": [{"name": name, "dtype": dtype,
                                      "channels_per_row": channels}
                                     for name, dtype, channels in value_fields],
        "estimated_channels": round(sample_channels * ratio),
        "estimated_time_steps_per_channel": round(sample_points / sample_channels, 2),
        "estimated_total_time_steps": round(sample_points * ratio),
    }


def lotsa_target_shape(feature: dict) -> tuple[int | None, int | None, str | None]:
    lengths = []
    while feature.get("_type") == "Sequence":
        lengths.append(feature.get("length"))
        feature = feature["feature"]
    dtype = feature.get("dtype")
    if len(lengths) == 1:
        channels = 1
    elif len(lengths) == 2:
        known = [length for length in lengths if isinstance(length, int) and length > 0]
        channels = min(known) if known else None
    else:
        channels = None
    return channels, VALUE_BYTES.get(dtype), dtype # type: ignore


def lotsa_info_estimate(repo: str, revision: str, info_path: str) -> dict:
    url = hf_hub_url(repo, info_path, repo_type="dataset", revision=revision)
    for attempt in range(5):
        response = requests.get(url, timeout=30)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            break
        if attempt == 4:
            response.raise_for_status()
        delay = min(float(response.headers.get("Retry-After", 2 ** attempt)), 30)
        time.sleep(delay)
    info = response.json()
    channels_per_row, value_bytes, dtype = lotsa_target_shape(info["features"]["target"])
    rows = sum(int(split["num_examples"]) for split in info["splits"].values())
    data_bytes = int(info.get("dataset_size") or info.get("size_in_bytes") or 0)
    channels = rows * channels_per_row if channels_per_row is not None else None
    points = round(data_bytes / value_bytes) if data_bytes and value_bytes else None
    return {
        "method": "dataset_info_examples_and_target_dtype",
        "metadata_file": info_path,
        "dataset_size_mb": mb(data_bytes),
        "rows": rows,
        "target_dtype": dtype,
        "channels_per_row": channels_per_row,
        "estimated_channels": channels,
        "estimated_time_steps_per_channel": round(points / channels, 2) if points and channels else None,
        "estimated_total_time_steps": points,
        "time_step_estimate_is_upper_bound": True,
    }


def inventory_remote(source: str, workers: int) -> dict:
    repo = REPOS[source]
    api = HfApi()
    revision = api.repo_info(repo, repo_type="dataset").sha
    groups: dict[str, list[RepoFile]] = defaultdict(list)
    folder_bytes: dict[str, int] = defaultdict(int)
    top_bytes: dict[str, int] = defaultdict(int)
    info_paths: dict[str, str] = {}
    all_bytes = 0
    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True):
        if not isinstance(entry, RepoFile):
            continue
        size = entry.size or 0
        all_bytes += size
        top_bytes[entry.path.split("/")[0]] += size
        parent = entry.path.rpartition("/")[0]
        folder_bytes[parent] += size
        if entry.path.endswith((".arrow", ".parquet")):
            groups[parent].append(entry)
        elif entry.path.endswith("/dataset_info.json"):
            info_paths[parent] = entry.path

    def inspect_directory(path: str, data_files: list[RepoFile]) -> dict:
        data_bytes = sum(item.size or 0 for item in data_files)
        row = {"path": path, "data_files": len(data_files),
               "directory_size_mb": mb(folder_bytes[path]),
               "data_size_mb": mb(data_bytes),
               "eligible_for_prepare": not bool(EVAL_NAMES.search(path)),
               "input": f"hf://datasets/{repo}/{path}"}
        try:
            if source == "chronos":
                parquet_files = [item for item in data_files if item.path.endswith(".parquet")]
                if not parquet_files:
                    raise ValueError("no Parquet files")
                row.update(first_parquet_estimate(repo, revision, parquet_files)) # type: ignore
            else:
                if path not in info_paths:
                    raise ValueError("dataset_info.json is missing")
                row.update(lotsa_info_estimate(repo, revision, info_paths[path])) # type: ignore
            row["status"] = "estimated"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_directory, path, files): path
                   for path, files in groups.items()}
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0 or len(rows) == len(groups):
                print(f"{source}: {len(rows)}/{len(groups)} directories", file=sys.stderr, flush=True)
    rows.sort(key=lambda row: row["path"])
    return {"repo": repo, "revision": revision,
            "repository_size_mb": mb(all_bytes),
            "top_level_sizes_mb": {key: mb(value) for key, value in sorted(top_bytes.items())},
            "directories": rows}


def inventory_utsd(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("input", "").rstrip("/").split("/")[-1] != "UTSD-12G":
        raise ValueError(f"{root} manifest is not for UTSD-12G")
    splits = {}
    for split in ("train", "validation"):
        total_bytes = 0
        total_points = 0
        total_rows = 0
        for shard in manifest["shards"][split]:
            path = root / shard["path"]
            metadata = pq.read_metadata(path)
            if metadata.num_rows != shard["rows"]:
                raise ValueError(f"manifest row count differs from {path}")
            total_rows += metadata.num_rows
            total_bytes += path.stat().st_size
            for group in range(metadata.num_row_groups):
                columns = [metadata.row_group(group).column(index)
                           for index in range(metadata.row_group(group).num_columns)]
                total_points += next(column.num_values for column in columns
                                     if column.path_in_schema == "values.list.element")
        splits[split] = {"shards": len(manifest["shards"][split]),
                         "channels": total_rows,
                         "time_steps": total_points,
                         "size_mb": mb(total_bytes)}
    if splits["train"]["channels"] != manifest["train_series"]:
        raise ValueError("train channel count differs from manifest")
    return {"input": manifest["input"], "revision": manifest.get("revision"),
            "source_files": manifest.get("source_files"),
            "total_channels": splits["train"]["channels"],
            "total_time_steps": sum(item["time_steps"] for item in splits.values()),
            "total_size_mb": round(sum(item["size_mb"] for item in splits.values()), 3),
            "splits": splits}


def select_directories(rows: list[dict], target: int) -> dict:
    """Approximate the closest channel count, keeping the smallest bytes per bucket."""
    candidates = [row for row in rows if row["eligible_for_prepare"]
                  and isinstance(row.get("estimated_channels"), int)
                  and row["estimated_channels"] > 0
                  and (row.get("estimated_time_steps_per_channel") or 0) >= 160]
    bucket_size = max(1, target // 1000)
    states: dict[int, tuple[int, float, tuple[str, ...]]] = {0: (0, 0.0, ())}
    for row in candidates:
        next_states = dict(states)
        for count, size, paths in states.values():
            new_count = count + row["estimated_channels"]
            if new_count > 2 * target:
                continue
            new_size = size + row["data_size_mb"]
            bucket = round(new_count / bucket_size)
            old = next_states.get(bucket)
            if old is None or (new_size, abs(new_count - bucket * bucket_size)) < (
                    old[1], abs(old[0] - bucket * bucket_size)):
                next_states[bucket] = (new_count, new_size, paths + (row["path"],))
        states = next_states
    count, size, paths = min(states.values(), key=lambda state: (abs(state[0] - target), state[1]))
    return {"method": "bucketed_subset_sum_minimizing_channel_gap_then_download_size",
            "minimum_estimated_time_steps_per_channel": 160,
            "target_channels": target, "estimated_channels": count,
            "channel_gap": count - target, "data_size_mb": round(size, 3),
            "estimated_total_time_steps": sum(
                row["estimated_total_time_steps"] for row in candidates
                if row["path"] in paths),
            "directories": list(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utsd", type=Path, default=Path("corpus/utsd"))
    parser.add_argument("--output", type=Path, default=Path("results/data_inventory.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "size_unit": "MB = 1,000,000 bytes",
              "assumptions": [
                  "Chronos estimates scale the first Parquet file's row and target-value counts by total Parquet bytes.",
                  "LOTSA channel counts use dataset_info split examples and fixed target channel dimension; unknown dimensions remain null.",
                  "LOTSA time-step estimates divide dataset_size by target dtype bytes and are upper bounds because metadata also uses space.",
                  "Sundial preprocessing can skip evaluation names and short or empty series; remote estimates do not predict all such skips.",
                  "Suggested selections require estimated mean series length of at least 160, but this does not guarantee every series survives preprocessing.",
              ],
              "utsd": inventory_utsd(args.utsd), "sources": {}}
    for source in REPOS:
        print(f"Inspecting {source}", file=sys.stderr, flush=True)
        result["sources"][source] = inventory_remote(source, args.workers)
        result["sources"][source]["suggested_selection"] = select_directories(
            result["sources"][source]["directories"], result["utsd"]["total_channels"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
