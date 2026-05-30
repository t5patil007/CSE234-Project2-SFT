#!/usr/bin/env python3
"""Project 2 data/schema diagnostics.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def schema_filename_for_db(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def load_schema_maps(schemas_dir: Path):
    schemas = {}
    for schema_path in schemas_dir.glob("*.json"):
        if schema_path.name == "_index.json":
            continue
        payload = load_json(schema_path)
        table_names = payload["table_names_original"]
        columns_by_table = defaultdict(set)
        for table_idx, column_name in payload["column_names_original"]:
            if table_idx == -1:
                continue
            table_name = table_names[table_idx]
            columns_by_table[table_name].add(column_name)
        schemas[payload["db_id"]] = {
            "tables": set(table_names),
            "columns_by_table": dict(columns_by_table),
        }
    return schemas


def normalize_link_dict(schema_links):
    if not isinstance(schema_links, dict):
        return {}
    cleaned = {}
    for table_name, cols in schema_links.items():
        if isinstance(cols, list):
            cleaned[str(table_name)] = [str(c) for c in cols]
        else:
            cleaned[str(table_name)] = []
    return cleaned


def analyze_split(split_name, rows, schemas, rare_limit=200):
    db_counter = Counter()
    table_count_per_q = []
    col_count_per_q = []
    table_only_count = 0
    any_empty_col_list_count = 0
    multi_table_count = 0
    missing_schema_count = 0
    question_char_lengths = []
    question_word_lengths = []

    invalid_table_refs = 0
    invalid_col_refs = 0
    rows_with_invalid_refs = 0

    table_freq = Counter()
    table_col_freq = Counter()
    per_db_stats = defaultdict(lambda: {"examples": 0, "table_links": 0, "column_links": 0})

    for row in rows:
        db_id = row["db_id"]
        question = str(row.get("question", ""))
        question_char_lengths.append(len(question))
        question_word_lengths.append(len(question.split()))
        db_counter[db_id] += 1
        per_db_stats[db_id]["examples"] += 1

        schema = schemas.get(db_id)
        if not schema:
            missing_schema_count += 1
            continue

        links = normalize_link_dict(row.get("schema_links", {}))
        tables = list(links.keys())
        table_count = len(tables)
        col_count = sum(len(cols) for cols in links.values())
        table_count_per_q.append(table_count)
        col_count_per_q.append(col_count)

        per_db_stats[db_id]["table_links"] += table_count
        per_db_stats[db_id]["column_links"] += col_count

        if table_count > 1:
            multi_table_count += 1

        if any(len(cols) == 0 for cols in links.values()):
            any_empty_col_list_count += 1
        if table_count > 0 and all(len(cols) == 0 for cols in links.values()):
            table_only_count += 1

        row_has_invalid = False
        for table_name, cols in links.items():
            table_freq[f"{db_id}.{table_name}"] += 1
            if table_name not in schema["tables"]:
                invalid_table_refs += 1
                row_has_invalid = True
                invalid_col_refs += len(cols)
                continue
            valid_cols = schema["columns_by_table"].get(table_name, set())
            for col_name in cols:
                table_col_freq[f"{db_id}.{table_name}.{col_name}"] += 1
                if col_name not in valid_cols:
                    invalid_col_refs += 1
                    row_has_invalid = True
        if row_has_invalid:
            rows_with_invalid_refs += 1

    total_examples = len(rows)
    avg_tables = mean(table_count_per_q) if table_count_per_q else 0.0
    avg_cols = mean(col_count_per_q) if col_count_per_q else 0.0
    avg_question_chars = mean(question_char_lengths) if question_char_lengths else 0.0
    avg_question_words = mean(question_word_lengths) if question_word_lengths else 0.0

    per_db = {}
    for db_id, stats in sorted(per_db_stats.items(), key=lambda item: item[0]):
        examples = stats["examples"]
        per_db[db_id] = {
            "examples": examples,
            "avg_tables_per_example": (stats["table_links"] / examples) if examples else 0.0,
            "avg_columns_per_example": (stats["column_links"] / examples) if examples else 0.0,
        }

    rare_tables = [
        {"table": key, "count": count}
        for key, count in sorted(table_freq.items(), key=lambda item: (item[1], item[0]))
        if count <= 2
    ]
    rare_table_columns = [
        {"table_column": key, "count": count}
        for key, count in sorted(table_col_freq.items(), key=lambda item: (item[1], item[0]))
        if count <= 2
    ]

    return {
        "split_name": split_name,
        "total_examples": total_examples,
        "unique_db_ids": len(db_counter),
        "db_distribution": dict(sorted(db_counter.items(), key=lambda item: (-item[1], item[0]))),
        "metrics": {
            "avg_tables_per_example": avg_tables,
            "avg_columns_per_example": avg_cols,
            "avg_question_chars": avg_question_chars,
            "avg_question_words": avg_question_words,
            "multi_table_example_rate": (multi_table_count / total_examples) if total_examples else 0.0,
            "table_only_example_rate": (table_only_count / total_examples) if total_examples else 0.0,
            "examples_with_any_empty_column_list_rate": (
                any_empty_col_list_count / total_examples if total_examples else 0.0
            ),
        },
        "schema_validation": {
            "missing_schema_count": missing_schema_count,
            "rows_with_invalid_refs": rows_with_invalid_refs,
            "invalid_table_refs": invalid_table_refs,
            "invalid_column_refs": invalid_col_refs,
        },
        "per_db_stats": per_db,
        "long_tail": {
            "rare_tables_leq2_count": len(rare_tables),
            "rare_tables_leq2_sample": rare_tables[:rare_limit],
            "rare_table_columns_leq2_count": len(rare_table_columns),
            "rare_table_columns_leq2_sample": rare_table_columns[:rare_limit],
            "top_tables": [
                {"table": key, "count": count} for key, count in table_freq.most_common(15)
            ],
            "top_table_columns": [
                {"table_column": key, "count": count} for key, count in table_col_freq.most_common(20)
            ],
        },
    }


def build_markdown_report(report):
    train = report["splits"]["train"]
    val = report["splits"]["validation"]
    underrepresented = report["cross_split"]["underrepresented_db_ids_leq6_train_examples"]
    sbodemo_modules = report["cross_split"]["sbodemous_train_counts"]

    lines = [
        "# Data Diagnostics Summary",
        "",
        "## Dataset shape",
        f"- Train examples: {train['total_examples']}",
        f"- Validation examples: {val['total_examples']}",
        f"- Train unique db_id count: {train['unique_db_ids']}",
        f"- Validation unique db_id count: {val['unique_db_ids']}",
        f"- Schema count: {report['schema_inventory']['num_schemas']}",
        f"- Avg tables/schema: {report['schema_inventory']['avg_tables']:.2f}",
        f"- Avg columns/schema: {report['schema_inventory']['avg_columns']:.2f}",
        f"- Max tables in a schema: {report['schema_inventory']['max_tables']}",
        f"- Max columns in a schema: {report['schema_inventory']['max_columns']}",
        "",
        "## Link complexity",
        f"- Train avg tables/example: {train['metrics']['avg_tables_per_example']:.3f}",
        f"- Train avg columns/example: {train['metrics']['avg_columns_per_example']:.3f}",
        f"- Train multi-table rate: {train['metrics']['multi_table_example_rate']:.3%}",
        f"- Train avg question chars: {train['metrics']['avg_question_chars']:.1f}",
        f"- Train avg question words: {train['metrics']['avg_question_words']:.1f}",
        f"- Validation avg tables/example: {val['metrics']['avg_tables_per_example']:.3f}",
        f"- Validation avg columns/example: {val['metrics']['avg_columns_per_example']:.3f}",
        f"- Validation multi-table rate: {val['metrics']['multi_table_example_rate']:.3%}",
        f"- Validation avg question chars: {val['metrics']['avg_question_chars']:.1f}",
        f"- Validation avg question words: {val['metrics']['avg_question_words']:.1f}",
        "",
        "## Wildcard / table-only signatures",
        (
            "- Train table-only rate (all linked tables have empty column lists): "
            f"{train['metrics']['table_only_example_rate']:.3%}"
        ),
        (
            "- Validation table-only rate (all linked tables have empty column lists): "
            f"{val['metrics']['table_only_example_rate']:.3%}"
        ),
        (
            "- Train examples with any empty column list: "
            f"{train['metrics']['examples_with_any_empty_column_list_rate']:.3%}"
        ),
        (
            "- Validation examples with any empty column list: "
            f"{val['metrics']['examples_with_any_empty_column_list_rate']:.3%}"
        ),
        "",
        "## Schema validity checks",
        f"- Train rows with invalid refs: {train['schema_validation']['rows_with_invalid_refs']}",
        f"- Validation rows with invalid refs: {val['schema_validation']['rows_with_invalid_refs']}",
        f"- Train missing-schema rows: {train['schema_validation']['missing_schema_count']}",
        f"- Validation missing-schema rows: {val['schema_validation']['missing_schema_count']}",
        "",
        "## Under-represented db_id in train (<=6 examples)",
        f"- Count: {len(underrepresented)}",
    ]

    for item in underrepresented:
        lines.append(f"- {item['db_id']}: {item['count']} examples")

    lines.extend(["", "## SBODemoUS module counts in train"])
    if sbodemo_modules:
        for item in sbodemo_modules:
            lines.append(f"- {item['db_id']}: {item['count']} examples")
    else:
        lines.append("- No SBODemoUS modules found in train split.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".", help="Path to Project2 directory")
    parser.add_argument("--output_dir", default="analysis", help="Directory to write diagnostics artifacts")
    parser.add_argument(
        "--rare_limit",
        type=int,
        default=200,
        help="Max rows to persist for each rare-identifier sample list in JSON output.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_json(root / "train.json")
    validation = load_json(root / "validation.json")
    schemas = load_schema_maps(root / "schemas")
    index_payload = load_json(root / "schemas" / "_index.json")

    train_report = analyze_split("train", train, schemas, rare_limit=args.rare_limit)
    val_report = analyze_split("validation", validation, schemas, rare_limit=args.rare_limit)

    train_db_counter = Counter(train_report["db_distribution"])
    underrepresented = [
        {"db_id": db_id, "count": count}
        for db_id, count in sorted(train_db_counter.items(), key=lambda item: (item[1], item[0]))
        if count <= 6
    ]
    sbodemo_counts = [
        {"db_id": db_id, "count": count}
        for db_id, count in sorted(train_db_counter.items(), key=lambda item: item[0])
        if db_id.startswith("SBODemoUS-")
    ]

    report = {
        "splits": {"train": train_report, "validation": val_report},
        "schema_inventory": {
            "num_schemas": len(index_payload),
            "max_tables": max(item["num_tables"] for item in index_payload) if index_payload else 0,
            "max_columns": max(item["num_columns"] for item in index_payload) if index_payload else 0,
            "avg_tables": mean(item["num_tables"] for item in index_payload) if index_payload else 0.0,
            "avg_columns": mean(item["num_columns"] for item in index_payload) if index_payload else 0.0,
        },
        "cross_split": {
            "underrepresented_db_ids_leq6_train_examples": underrepresented,
            "sbodemous_train_counts": sbodemo_counts,
        },
    }

    json_path = output_dir / "data_diagnostics.json"
    md_path = output_dir / "data_diagnostics.md"

    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    with md_path.open("w") as f:
        f.write(build_markdown_report(report))

if __name__ == "__main__":
    main()
