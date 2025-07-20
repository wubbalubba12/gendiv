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
        base = str(col).strip() if not pd.isna(col) else "Unnamed"
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

# === Column grouping by base question name ===
question_variants = defaultdict(list)
for col in df.columns:
    base = re.sub(r"\s*\.\d+$", "", col).strip()
    base = re.sub(r"\s+", " ", base)
    question_variants[base].append(col)

# Manual override for the academic task question
def assign_special_case_phases(df):
    # Match deduplicated base names
    base_name = "Were you able to stay on top of your academic tasks during this time? (Question inspired by Ralph et al. (2020))"
    
    task_question_cols = [col for col in df.columns if base_name in col]

    if len(task_question_cols) < 3:
        raise ValueError(f"Expected 3 columns for special case, found {len(task_question_cols)}: {task_question_cols}")

    # Use column positions to ensure consistent order
    sorted_cols = sorted(task_question_cols, key=lambda c: df.columns.get_indexer_for([c])[0])

    # Assign phases manually
    phase_map = {
        sorted_cols[0]: "before",
        sorted_cols[1]: "during",
        sorted_cols[2]: "after"
    }
    return phase_map

# === Phase Assignment Function (Updated) ===
def get_phase_by_index(col, variants, df):
    phase_order = ['before', 'during', 'after']
    variants_sorted = sorted(variants, key=lambda c:df.columns.get_indexer_for([c])[0])
    index = variants_sorted.index(col)
    if len(variants_sorted) >= 3:
        return phase_order[index]
    elif len(variants_sorted) == 2:
        return ['before', 'after'][index]
    elif len(variants_sorted) == 1:
        return 'yes/no'
    else:
        return 'unknown'

# === Identify Open Text Columns ===
text_questions = {}
skipped_texts = []

with open(TEXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
    for base, variants in question_variants.items():
        for col in sorted(variants, key=lambda c: df.columns.get_indexer_for([c])[0]):
            phase = get_phase_by_index(col, variants, df)

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
                else:
                    skipped_texts.append(base)

    for question, responses in text_questions.items():
        f.write(f"\n=== {question} ===\n")
        for resp in responses:
            f.write(f"- {resp}\n")

print(f"\n✅ Exported {len(text_questions)} open text questions to text_responses.txt")

if skipped_texts:
    print("\n⚠️ Skipped short text questions:")
    for q in skipped_texts:
        print("-", q)

# === Identify and Normalize Likert + Yes/No Questions ===
likert_pattern = re.compile(r"^\d+\s*-\s*.+")
yesno_pattern = re.compile(r"^(yes|no)$", re.IGNORECASE)

likert_rows = []

special_phase_map = assign_special_case_phases(df)  # Add this before the loops

for base, variants in question_variants.items():
    for col in sorted(variants, key=lambda c: df.columns.get_indexer_for([c])[0]):
        if col in special_phase_map:
            phase = special_phase_map[col]
        else:
            phase = get_phase_by_index(col, variants, df)

        if any(s in col for s in ["[Punktzahl]", "Feedback", "Unnamed", "Gesamtpunktzahl", "Zeitstempel"]):
            continue

        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.astype(str).str.strip()

        likert_ratio = series.str.match(likert_pattern).mean()
        yesno_ratio = series.str.match(yesno_pattern).mean()
        is_likert = likert_ratio > 0.6
        is_yesno = yesno_ratio > 0.8

        if not (is_likert or is_yesno):
            unique_vals = series.dropna().unique()
            if 1 < len(unique_vals) <= 6:
                is_fallback = True
            else:
                continue
        else:
            is_fallback = False

        for idx, val in series.items():
            val = val.strip()
            if val == "" or val.lower() == "nan":
                continue
            response = val.capitalize() if is_yesno else val.strip()
            likert_rows.append({
                "participant": idx,
                "question": base,
                "original_column": col,
                "phase": phase,
                "response": response
            })

print(f"\n✅ Extracted {len(set([row['question'] for row in likert_rows]))} Likert/Yes/No questions")

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

# === Save Likert/YesNo responses ===
likert_df = pd.DataFrame(likert_rows)
likert_df.to_csv(LIKERT_OUTPUT_FILE, index=False)

print(f"\n✅ Normalized Likert responses saved to {LIKERT_OUTPUT_FILE.name} with shape {likert_df.shape}")
