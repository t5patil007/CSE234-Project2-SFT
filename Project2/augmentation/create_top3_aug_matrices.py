#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


TOP3_CONFIG_IDS = [
    "w2_cfg07_lora_qwen15b_minimal_lr2e5_a100",
    "w2_cfg05_lora_prompt_filtered_strict_a100",
    "w2_cfg10_fullft_qwen05b_filtered_lr1e5_a100",
]

STRATEGIES = {
    "ctrl_noaug": "ctrl_noaug",
    "augA_local_sql": "augA_local_sql",
    "augB_targeted_hard": "augB_targeted_hard",
    "augC_hybrid_small": "augC_hybrid_small",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_matrix",
        default="configs/config_matrix_wave2_a100_missing_weights_rerun.json",
    )
    parser.add_argument("--out_dir", default="configs")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    src_path = (project_root / args.source_matrix).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    source_rows = load_json(src_path)
    by_id = {r["config_id"]: r for r in source_rows}
    missing = [cid for cid in TOP3_CONFIG_IDS if cid not in by_id]
    if missing:
        raise ValueError(f"Missing top3 config ids in source matrix: {missing}")

    for strategy, suffix in STRATEGIES.items():
        rows = []
        for cid in TOP3_CONFIG_IDS:
            row = copy.deepcopy(by_id[cid])
            row["parent_config_id"] = row["config_id"]
            row["augmentation_strategy"] = strategy
            row["config_id"] = f"{row['config_id']}_{suffix}"
            rows.append(row)
        out_path = out_dir / f"config_matrix_top3_{strategy}.json"
        save_json(out_path, rows)
        print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
