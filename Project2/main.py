import argparse
import json
import os


def load_schema(db_id, schemas_dir):
    fname = db_id.replace(" ", "_").replace("/", "_") + ".json"
    path = os.path.join(schemas_dir, fname)
    with open(path) as f:
        payload = json.load(f)
    table_names = payload["table_names_original"]
    lc_tables = {t.lower(): t for t in table_names}
    lc_cols = {t: {} for t in table_names}
    for tidx, cname in payload["column_names_original"]:
        if tidx == -1:
            continue
        table_name = table_names[tidx]
        lc_cols[table_name][cname.lower()] = cname
    return {"lc_tables": lc_tables, "lc_cols": lc_cols}


def build_prompt(question, db_id, schema_info):
    tables = sorted(schema_info["lc_tables"].values())
    schema_text = " ; ".join(tables)
    return f"db_id={db_id}\nquestion={question}\nschema={schema_text}\noutput=json"


def normalize_links(raw_links, schema_info):
    if not isinstance(raw_links, dict):
        return {}
    normalized = {}
    lc_tables = schema_info["lc_tables"]
    lc_cols = schema_info["lc_cols"]
    for table_name, cols in raw_links.items():
        canonical_table = lc_tables.get(str(table_name).lower())
        if not canonical_table:
            continue
        col_values = cols if isinstance(cols, list) else []
        seen = set()
        clean_cols = []
        for col_name in col_values:
            canonical_col = lc_cols.get(canonical_table, {}).get(str(col_name).lower())
            if canonical_col and canonical_col not in seen:
                seen.add(canonical_col)
                clean_cols.append(canonical_col)
        normalized[canonical_table] = sorted(clean_cols)
    return {t: normalized[t] for t in sorted(normalized)}


def predict_schema_links(question, db_id, schema_info):
    _ = build_prompt(question, db_id, schema_info)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")
    args = ap.parse_args()

    with open(args.input) as f:
        items = json.load(f)

    schema_cache = {}
    preds = []
    for item in items:
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(db_id, args.schemas_dir)
        schema_info = schema_cache[db_id]
        raw_links = predict_schema_links(item["question"], db_id, schema_info)
        links = normalize_links(raw_links, schema_info)
        preds.append({"question_id": item["question_id"], "schema_links": links})

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()
