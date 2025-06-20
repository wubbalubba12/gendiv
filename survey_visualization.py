import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

file_path = r"C:\Users\nikkh\Desktop\Survey_vis.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.extract(r"^([^[]+)", expand=False).str.strip()
df = df.dropna(axis=1, how='all')

likert_map = {
    "1 - Never": 1,
    "2 - Rarely": 2,
    "3 - Sometimes": 3,
    "4 - Often": 4,
    "5 - Always": 5
}

likert_cols = []
text_cols = []
categorical_cols = []

for col in df.columns:
    sample = df[col].dropna().astype(str)
    if sample.empty:
        continue
    if sample.str.match(r"^[1-5] - ").all():
        df[col + "_score"] = df[col].map(likert_map)
        likert_cols.append(col)
    elif sample.nunique() < 20:
        categorical_cols.append(col)
    else:
        text_cols.append(col)

print("\n=== LIKERT SCALE QUESTIONS ===")
print(likert_cols)

print("\n=== CATEGORICAL QUESTIONS ===")
print(categorical_cols)

print("\n=== OPEN-TEXT QUESTIONS ===")
print(text_cols)

if likert_cols:
    print("\n=== LIKERT SCALE SUMMARY ===")
    print(df[[col + "_score" for col in likert_cols]].describe())

    print("\n=== LIKERT SCALE CORRELATION ===")
    print(df[[col + "_score" for col in likert_cols]].corr())

    for col in likert_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df, order=likert_map.keys())
        plt.title(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if categorical_cols:
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(col)
        plt.tight_layout()
        plt.show()

if text_cols:
    for col in text_cols:
        print(f"\n=== SAMPLE TEXT RESPONSES for: {col} ===")
        print(df[col].dropna().sample(min(5, df[col].dropna().shape[0])))

        text = " ".join(df[col].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for: {col}")
        plt.tight_layout()
        plt.show()
