import pandas as pd
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Survey.csv"

# Deduplicate columns (in case of identical questions)
def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if pd.isna(col):
            base = "Unnamed"
        else:
            base = str(col).strip()
        if base not in seen:
            seen[base] = 0
            new_cols.append(base)
        else:
            seen[base] += 1
            new_cols.append(f"{base}.{seen[base]}")
    return new_cols

def clean_column_headers(df):
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df.columns = deduplicate_columns(df.columns)
    return df

def clean_responses(series):
    return series.dropna().astype(str).map(str.strip).replace('--', pd.NA).dropna()

def smart_group_questions(df):
    groups = {}
    for col in df.columns:
        if col.startswith("Unnamed"):
            continue
        base = re.sub(r'\s*\[.*?\]$', '', col).strip()
        if base not in groups:
            groups[base] = {}
        if '[Feedback]' in col:
            groups[base]['feedback'] = df[col]
        elif '[Punktzahl]' in col:
            groups[base]['score'] = df[col]
        else:
            groups[base]['text'] = df[col]
    return groups

def load_data():
    df = pd.read_csv(DATA_FILE, header=0, skiprows=[1])
    df = df.dropna(axis=1, how="all")
    df = clean_column_headers(df)
    return df