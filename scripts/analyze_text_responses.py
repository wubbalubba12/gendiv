import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def load_responses(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        responses = f.read().split('\n')
    responses = [r.strip() for r in responses if r.strip()]
    return responses

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stopwords = set([
        'the', 'and', 'to', 'of', 'a', 'in', 'i', 'was', 'it', 'for', 'that', 'is',
        'on', 'my', 'with', 'at', 'had', 'but', 'not', 'as', 'be', 'have', 'this',
        'so', 'we', 'were', 'just', 'me', 'they', 'from', 'by', 'or', 'when', 'you'
    ])
    return [t for t in tokens if t not in stopwords]

def get_word_frequencies(responses):
    all_tokens = []
    for response in responses:
        all_tokens.extend(preprocess(response))
    return Counter(all_tokens)

def extract_tfidf_keywords(responses, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(responses)
    tfidf_scores = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked_terms = sorted(zip(terms, tfidf_scores), key=lambda x: -x[1])
    return ranked_terms[:top_n]

def generate_wordcloud(freqs):
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(freqs)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    filepath = 'data/text_responses.txt'
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return

    responses = load_responses(filepath)
    print(f"\nTotal Responses: {len(responses)}")

    word_freqs = get_word_frequencies(responses)
    print("\nTop 10 most common words:")
    for word, count in word_freqs.most_common(10):
        print(f"{word}: {count}")

    tfidf_keywords = extract_tfidf_keywords(responses)
    print("\nTop 10 keywords (TF-IDF):")
    for word, score in tfidf_keywords:
        print(f"{word}: {score:.2f}")

    generate_wordcloud(word_freqs)

if __name__ == '__main__':
    main()
