import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from read_data import load_data, smart_group_questions
import re
import warnings
warnings.filterwarnings("ignore")

def safe_filename(text):
    import os
    name, ext = os.path.splitext(text)
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    name = name.replace('\n', ' ')
    name = re.sub(r'\s+', '_', name.strip())
    return name[:80] + ext

# Paths
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TEXT_OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "text_responses.txt"

# Load and group data
df = load_data()
grouped = smart_group_questions(df)

# Define Likert scales
likert_maps = {
    "standard": {
        "1 - Never": 1, "2 - Rarely": 2, "3 - Sometimes": 3, "4 - Often": 4, "5 - Always": 5
    },
    "agreement": {
        "1 - Strongly disagree": 1, "2 - Disagree": 2, "3 - Neither disagree or agree": 3,
        "4 - Agree": 4, "5 - Strongly agree": 5
    },
    "intensity": {
        "1 - Very low": 1, "2 - Low": 2, "3 - Average": 3, "4 - High": 4, "5 - Very high": 5
    },
    "change": {
        "1 - Increased": 1, "2 - Stayed the same": 2, "3 - Decreased": 3
    }
}

valid_text_cols = []

# Process grouped questions
for question, parts in grouped.items():
    # Likert-style score questions
    if 'score' in parts:
        series = parts['score'].dropna().astype(str)
        if series.empty:
            continue

        matched = False
        for scale_map in likert_maps.values():
            mask = series.isin(scale_map.keys())
            match_ratio = mask.mean()  # Fix: avoid ambiguous Series

            if match_ratio > 0.6:
                # Plot Likert distribution
                plt.figure(figsize=(8, 4))
                sns.countplot(x=series, order=list(scale_map.keys()))
                plt.title(question)
                plt.xticks(rotation=45)
                plt.tight_layout()
                filename = safe_filename(question + "_likert.png")
                plt.savefig(PLOTS_DIR / filename)
                plt.close()
                matched = True
                break
        if not matched:
            print(f"Skipped Likert plot: {question} – unknown scale")

    # Export text responses
    if 'text' in parts:
        responses = parts['text'].dropna().astype(str).map(str.strip)
        responses = responses[responses != ""]
        if not responses.empty:
            valid_text_cols.append(question)
            with open(TEXT_OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n=== {question} ===\n")
                for resp in responses:
                    f.write(f"- {resp}\n")

print(f"\nDetected {len(valid_text_cols)} text-based questions:\n{valid_text_cols}")
