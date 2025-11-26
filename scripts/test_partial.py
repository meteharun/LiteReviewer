import os
import json
import pandas as pd

EXPERIMENTS_DIR = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\experiments"
JSONL_PATH = r"C:\Users\msi-nb\Desktop\AIS\LiteReviewer\dataset\experiment.jsonl"

def has_docstring(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return ('"""' in text) or ("'''" in text)

def load_has_docstr_flags(jsonl_path: str):
    flags = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            diff = row.get("diff_hunk", "")
            flags.append(has_docstring(diff))
    return flags

def insert_after_type(df: pd.DataFrame, col_name: str, values):
    if "type" not in df.columns:
        raise ValueError("No 'type' column found in CSV.")
    if len(values) != len(df):
        raise ValueError(f"Length mismatch: flags={len(values)} vs df={len(df)}")

    df[col_name] = values

    # reorder so col_name is right after 'type'
    cols = list(df.columns)
    cols.remove(col_name)
    type_idx = cols.index("type")
    cols.insert(type_idx + 1, col_name)
    return df[cols]

def main():
    flags = load_has_docstr_flags(JSONL_PATH)

    csv_files = [
        os.path.join(EXPERIMENTS_DIR, f)
        for f in os.listdir(EXPERIMENTS_DIR)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        print(f"[WARN] No CSV files found in {EXPERIMENTS_DIR}")
        return

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)

            # add column after "type"
            df = insert_after_type(df, "has_docstr", flags)

            # save inplace
            df.to_csv(csv_path, index=False)
            print(f"[OK] Updated: {csv_path}")

        except Exception as e:
            print(f"[SKIP] {csv_path} -> {e}")

if __name__ == "__main__":
    main()
