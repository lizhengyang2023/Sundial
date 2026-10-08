"""Select Chronos and LOTSA directories by estimated time steps.

Run from the repository root:
    python src/utils/select_time_steps.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json


TARGET_TIME_STEPS = 850_000_000


def adjusted_steps(estimate: int) -> int:
    """Round 75% of an integer estimate without floating-point arithmetic."""
    return (estimate * 3 + 2) // 4


def report_source(source: dict, target: int) -> dict:
    directories = []
    for row in source["directories"]:
        estimate = row.get("estimated_total_time_steps")
        channels = row.get("estimated_channels")
        directories.append({
            "path": row["path"],
            "estimated_total_time_steps": estimate,
            "assumed_actual_time_steps_75pct": adjusted_steps(estimate) if estimate is not None else None,
            "estimated_channels": channels,
            "size_mb": row.get("directory_size_mb", row.get("data_size_mb")),
            "eligible_for_prepare": row["eligible_for_prepare"],
            "status": row["status"],
        })
    directories.sort(key=lambda row: (
        row["estimated_total_time_steps"] is None,
        row["estimated_total_time_steps"] or 0,
        row["path"],
    ))

    candidates = [row for row in directories
                  if row["eligible_for_prepare"]
                  and row["status"] == "estimated"
                  and row["estimated_total_time_steps"] is not None
                  and row["estimated_channels"] is not None]
    selected = []
    cumulative = 0
    for row in candidates:
        before = cumulative
        cumulative += row["assumed_actual_time_steps_75pct"]
        selected.append({
            "path": row["path"],
            "estimated_total_time_steps": row["estimated_total_time_steps"],
            "assumed_actual_time_steps_75pct": row["assumed_actual_time_steps_75pct"],
            "estimated_channels": row["estimated_channels"],
            "size_mb": row["size_mb"],
            "cumulative_assumed_actual_time_steps": cumulative,
        })
        if cumulative >= target:
            if selected[:-1] and abs(before - target) <= abs(cumulative - target):
                selected.pop()
                cumulative = before
            break

    next_candidate = candidates[len(selected)] if len(selected) < len(candidates) else None
    return {
        "repository": source["repo"],
        "revision": source["revision"],
        "directories_sorted_by_estimated_time_steps": directories,
        "selection": {
            "method": "ascending_estimated_steps_accumulate_then_choose_closer_side_of_target",
            "eligible_directories": len(candidates),
            "selected_directories": selected,
            "selected_paths": [row["path"] for row in selected],
            "estimated_channels": sum(row["estimated_channels"] for row in selected),
            "estimated_total_time_steps": sum(row["estimated_total_time_steps"] for row in selected),
            "assumed_actual_time_steps_75pct": cumulative,
            "difference_from_target": cumulative - target,
            "difference_pct": round(100 * (cumulative - target) / target, 3),
            "selected_size_mb": round(sum(row["size_mb"] for row in selected), 3),
            "next_directory_if_added": next_candidate["path"] if next_candidate else None,
            "assumed_actual_steps_if_next_added": (
                cumulative + next_candidate["assumed_actual_time_steps_75pct"]
                if next_candidate else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("results/data_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("results/time_step_selection.json"))
    parser.add_argument("--target", type=int, default=TARGET_TIME_STEPS)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be positive")

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_file": str(args.inventory),
        "utsd_reference_time_steps": inventory["utsd"]["total_time_steps"],
        "target_time_steps": args.target,
        "assumed_actual_fraction": 0.75,
        "selection_note": (
            "The 75% factor is applied to both sources as requested. "
            "Directories excluded by prepare's evaluation-name filter are listed but not selected. "
            "Estimates do not account for individual series discarded during conversion."
        ),
        "sources": {
            name: report_source(inventory["sources"][name], args.target)
            for name in ("chronos", "lotsa")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    for name, source in result["sources"].items():
        selection = source["selection"]
        print(f"{name}: {len(selection['selected_paths'])} directories, "
              f"{selection['assumed_actual_time_steps_75pct']:,} adjusted steps, "
              f"{selection['estimated_channels']:,} estimated channels, "
              f"{selection['difference_pct']:+.3f}% from target")
    print(args.output)


if __name__ == "__main__":
    main()
