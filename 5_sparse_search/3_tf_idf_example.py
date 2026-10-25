import math
from collections import Counter

documents = [
    "python python aws",
    "python java",
    "python fastapi"
]


# Step 1: Tokenize documents
tokenized_docs = [
    doc.lower().split()
    for doc in documents
]


# Step 2: Build vocabulary
vocabulary = sorted(set(
    word
    for doc in tokenized_docs
    for word in doc
))

# Step 3: Calculate document frequency
N = len(documents)


df = {
    word: sum(word in doc for doc in tokenized_docs)
    for word in vocabulary
}


# Step 4: Calculate IDF
idf = {
    word: math.log((N + 1) / (df[word] +1)) + 1
    for word in vocabulary
}


# Step 5: Calculate TF-IDF vectors
def calculate_tfidf(tokens):
    counts = Counter(tokens)
    total = len(tokens)

    return {
        word: (counts[word] / total) * idf[word]
        for word in vocabulary
    }


for i, tokens in enumerate(tokenized_docs):
    print(f"\nDocument {i + 1}")
    print(calculate_tfidf(tokens))
