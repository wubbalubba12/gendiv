import os
import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_responses(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        responses = f.read().split('\n')
    return [r.strip() for r in responses if r.strip()]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stops = set(stopwords.words('english'))
    return [t for t in tokens if t not in stops and len(t) > 2]

def main():
    filepath = 'data/text_responses.txt'
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        return

    documents = load_responses(filepath)
    processed_docs = [preprocess(doc) for doc in documents]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"Topic {idx + 1}: {topic}")

if __name__ == '__main__':
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    main()
