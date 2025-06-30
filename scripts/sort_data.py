import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

# === File Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Survey.csv"
TEXT_OUTPUT_FILE = BASE_DIR / "data" / "text_responses.txt"
LIKERT_OUTPUT_FILE = BASE_DIR / "data" / "likert_responses.csv"
PARTICIPANT_INFO_FILE = BASE_DIR / "data" / "participant_info.csv"

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

# === Load & Preprocess ===
df_survey = pd.read_csv(DATA_FILE, header=0, skiprows=[1])
cleaned_columns = df_survey.columns.str.replace(r"\s+", " ", regex=True).str.strip()
df_survey.columns = deduplicate_columns(cleaned_columns)
df = df_survey.dropna(axis=1, how="all")

# === Gruppiere Spaltenvarianten für Phasenzuordnung ===
question_variants = defaultdict(list)
for col in df.columns:
    base = re.sub(r"\.\d+$", "", col).strip()
    question_variants[base].append(col)

def get_phase_by_index(col, variants):
    index = variants.index(col)
    if len(variants) == 3:
        return ['before', 'during', 'after'][index]
    elif len(variants) == 2:
        return ['before', 'after'][index]
    else:
        return 'unspecified'

# === Identify Open Text Columns ===
text_questions = {}

with open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    for base, variants in question_variants.items():
        sorted_variants = sorted(variants)
        for col in sorted_variants:
            phase = get_phase_by_index(col, sorted_variants)

            if any(s in col for s in ["[Punktzahl]", "[Feedback]", "Unnamed", "Gesamtpunktzahl", "Zeitstempel"]):
                continue

            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.dropna().astype(str).map(str.strip)
            responses = series[series != ""]

            if responses.empty:
                continue

            if "(You can answer in full sentences or bullet points – whichever is easier for you.)" in base:
                if len(responses) >= 1 and responses.str.len().max() > 10:
                    question_key = f"{base} [{phase}]"
                    text_questions[question_key] = responses.tolist()

    # Write responses
    for question, responses in text_questions.items():
        f.write(f"\n=== {question} ===\n")
        for resp in responses:
            f.write(f"- {resp}\n")

print(f"\n✅ Exported {len(text_questions)} open text questions to text_responses.txt")

# === Identify and Normalize Likert + Yes/No Questions ===
likert_pattern = re.compile(r"^\d+ - ")
yesno_pattern = re.compile(r"^(Yes|No)$", re.IGNORECASE)

likert_rows = []

for base, variants in question_variants.items():
    sorted_variants = sorted(variants)
    for col in sorted_variants:
        phase = get_phase_by_index(col, sorted_variants)

        if any(s in col for s in ["[Punktzahl]", "[Feedback]", "Unnamed"]):
            continue

        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna().astype(str).str.strip()

        is_likert = series.str.match(likert_pattern).mean() > 0.6
        is_yesno = series.str.match(yesno_pattern).mean() > 0.8

        if not (is_likert or is_yesno):
            continue

        for i, (idx, val) in enumerate(series.items(), start=1):
            response = val.capitalize() if is_yesno else val
            likert_rows.append({
                "participant": i,
                "question": base,
                "phase": phase,
                "response": response
            })

# === Extract Participant Info ===
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

participant_cols = [col for col in df.columns if col in participant_questions]

participant_df = df[participant_cols].copy()
participant_df.to_csv(PARTICIPANT_INFO_FILE, index=False)

print(f"✅ Saved participant demographic info to {PARTICIPANT_INFO_FILE.name} with shape {participant_df.shape}")

# === Save Likert/YesNo responses to CSV ===
likert_df = pd.DataFrame(likert_rows)
likert_df.to_csv(LIKERT_OUTPUT_FILE, index=False)

print(f"\n✅ Normalized Likert responses saved to likert_responses.csv with shape {likert_df.shape}")
