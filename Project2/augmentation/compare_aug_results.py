#!/usr/bin/env python3
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


STRATEGY_SUFFIXES = [
    "ctrl_noaug",
    "augA_local_sql",
    "augB_targeted_hard",
    "augC_hybrid_small",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_glob",
        action="append",
        default=[],
        help="Glob(s) to summary.csv files (can pass multiple).",
    )
    parser.add_argument("--out_csv", default="augmentation/analysis/aug_strategy_compare.csv")
    parser.add_argument("--out_json", default="augmentation/analysis/aug_strategy_compare.json")
    return parser.parse_args()


def parse_score(row):
    for key in ("leaderboard_score", "score", "weighted_f1"):
        if key in row and row[key] not in ("", None):
            try:
                return float(row[key])
            except ValueError:
                pass
    return None


def parse_config_id(row):
    candidate_keys = ["config_id", "label", "run_label", "artifact_name"]
    raw = None
    for key in candidate_keys:
        if key in row and row[key]:
            raw = str(row[key])
            break
    if raw is None:
        return None

    clean = raw
    if "_adapter_" in clean:
        clean = clean.split("_adapter_")[0]
    if "_checkpoint_" in clean:
        clean = clean.split("_checkpoint_")[0]
    return clean


def split_strategy(config_id):
    for suffix in STRATEGY_SUFFIXES:
        token = f"_{suffix}"
        if config_id.endswith(token):
            return config_id[: -len(token)], suffix
    return config_id, "unknown"


def main():
    args = parse_args()
    patterns = args.summary_glob or [
        "logs/**/summary.csv",
        "/root/logs/**/summary.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        raise ValueError("No summary.csv files found. Pass --summary_glob with valid patterns.")

    rows = []
    for path in files:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = parse_score(row)
                config_id = parse_config_id(row)
                if score is None or not config_id:
                    continue
                base_id, strategy = split_strategy(config_id)
                rows.append(
                    {
                        "source_summary_csv": path,
                        "config_id": config_id,
                        "base_config_id": base_id,
                        "strategy": strategy,
                        "leaderboard_score": score,
                    }
                )

    if not rows:
        raise ValueError("No parseable rows found in supplied summary files.")

    best_by_base_strategy = {}
    for row in rows:
        key = (row["base_config_id"], row["strategy"])
        prev = best_by_base_strategy.get(key)
        if prev is None or row["leaderboard_score"] > prev["leaderboard_score"]:
            best_by_base_strategy[key] = row

    strategy_summary = defaultdict(list)
    for (base_id, strategy), row in best_by_base_strategy.items():
        strategy_summary[strategy].append((base_id, row["leaderboard_score"]))

    ctrl_by_base = {
        base_id: score for base_id, score in strategy_summary.get("ctrl_noaug", [])
    }

    output_rows = []
    for strategy, pairs in sorted(strategy_summary.items()):
        for base_id, score in sorted(pairs):
            ctrl = ctrl_by_base.get(base_id)
            delta = None if ctrl is None else score - ctrl
            output_rows.append(
                {
                    "base_config_id": base_id,
                    "strategy": strategy,
                    "leaderboard_score": round(score, 6),
                    "delta_vs_ctrl_noaug": None if delta is None else round(delta, 6),
                }
            )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_config_id",
                "strategy",
                "leaderboard_score",
                "delta_vs_ctrl_noaug",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    with open(out_json, "w") as f:
        json.dump(
            {
                "summary_files_scanned": files,
                "rows": output_rows,
            },
            f,
            indent=2,
        )

    print(f"Wrote comparison CSV:  {out_csv}")
    print(f"Wrote comparison JSON: {out_json}")


if __name__ == "__main__":
    main()
