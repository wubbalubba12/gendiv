import pandas as pd
import re
from pathlib import Path

# === File Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Survey.csv"
TEXT_OUTPUT_FILE = BASE_DIR / "data" / "text_responses.txt"
LIKERT_OUTPUT_FILE = BASE_DIR / "data" / "likert_responses.csv"
PARTICIPANT_INFO_FILE = BASE_DIR / "data" / "participant_info.csv"

# === Load & Preprocess ===
df = pd.read_csv(DATA_FILE, header=0, skiprows=[1])
df = df.dropna(axis=1, how="all")
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# === Identify Open Text Columns ===
text_questions = {}
phase_map = {"0": "before", "1": "during", "2": "after"}

with open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    for col in df.columns:
        if any(s in col for s in ["[Punktzahl]", "[Feedback]", "Unnamed", "Gesamtpunktzahl", "Zeitstempel"]):
            continue

        # Extract base column (without .0, .1, .2 suffix)
        suffix_match = re.search(r"\.(\d+)$", col)
        base_col = re.sub(r"\.\d+$", "", col).strip()

        if suffix_match:
            phase = phase_map.get(suffix_match.group(1), "during")
        else:
            # If other versions with suffix exist, this is likely "before"
            possible_variants = [c for c in df.columns if c.startswith(base_col + ".")]
            phase = "before" if possible_variants else "unspecified"

        # Prepare Series
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna().astype(str).map(str.strip)
        responses = series[series != ""]

        if responses.empty:
            continue

        unique_vals = responses.unique()
        avg_len = pd.Series(unique_vals).str.len().mean()

        # Only include questions explicitly marked as open-text
        if "(You can answer in full sentences or bullet points – whichever is easier for you.)" in base_col:
            if len(responses) >= 1 and responses.str.len().max() > 10:
                question_key = f"{base_col} [{phase}]"
                text_questions[question_key] = responses.tolist()

    # Now write results
    for question, responses in text_questions.items():
        f.write(f"\n=== {question} ===\n")
        for resp in responses:
            f.write(f"- {resp}\n")

print(f"\n✅ Exported {len(text_questions)} open text questions to text_responses.txt")

# === Identify and Normalize Likert + Yes/No Questions ===
likert_pattern = re.compile(r"^\d+ - ")
yesno_pattern = re.compile(r"^(Yes|No)$", re.IGNORECASE)

likert_rows = []

for col in df.columns:
    if any(s in col for s in ["[Punktzahl]", "[Feedback]", "Unnamed"]):
        continue

    series = df[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = series.dropna().astype(str).str.strip()

    # Check if Likert-style or Yes/No question
    is_likert = series.str.match(likert_pattern).mean() > 0.6
    is_yesno = series.str.match(yesno_pattern).mean() > 0.8

    if not (is_likert or is_yesno):
        continue  # Skip anything else

    # Extract base question
    suffix_match = re.search(r"\.(\d+)$", col)
    base_q = re.sub(r"\.\d+$", "", col).strip()

    # Infer phase
    if suffix_match:
        phase = {
            "0": "before",
            "1": "during",
            "2": "after"
        }.get(suffix_match.group(1), "during")
    else:
        possible_variants = [c for c in df.columns if c.startswith(base_q + ".")]
        phase = "before" if possible_variants else "unspecified"

    # Store responses
    for i, (idx, val) in enumerate(series.items(), start=1):
        response = val.capitalize() if is_yesno else val
        likert_rows.append({
            "participant": i,
            "question": base_q,
            "phase": phase,
            "response": response
        })

# Exact column names to keep (match from original CSV)
participant_questions = [
    "Are you currently enrolled in a university degree program?",
    "Which degree program are you currently pursuing? (e.g., B.Sc. Computer Science)",
    "What is your current year of study?",
    "Did you have a part-time job related to IT or software development during the pandemic (2020–2022)?",
    "What is your current age?",
    "With which gender do you identify?",
    "Did you primarily attend university remotely or in person during the pandemic?",
    "Where were you primarily located during the COVID-19 pandemic? (Please specify the country you spent most of your time in during this period.)",
    "Would you be interested in participating in a follow-up interview or study? If yes, please leave your email address below (your email will be stored separately and not linked to your responses):"
]

# Filter out unwanted "[Punktzahl]" versions
participant_cols = [col for col in df.columns if col in participant_questions]

# Extract and save
participant_df = df[participant_cols].copy()
participant_df.to_csv(PARTICIPANT_INFO_FILE, index=False)

print(f"✅ Saved participant demographic info to {PARTICIPANT_INFO_FILE.name} with shape {participant_df.shape}")

# === Save to CSV for analysis ===
likert_df = pd.DataFrame(likert_rows)
likert_df.to_csv(LIKERT_OUTPUT_FILE, index=False)

print(f"\n✅ Normalized Likert responses saved to likert_responses.csv with shape {likert_df.shape}")