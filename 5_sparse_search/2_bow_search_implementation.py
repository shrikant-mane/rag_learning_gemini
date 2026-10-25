
from collections import Counter

documents = [
    "python fastapi backend development",
    "java spring backend development",
    "python machine learning python"
]

# Create vocabulary
vocabulary = sorted(set(
    word
    for doc in documents
    for word in doc.lower().split()
))

def encode(text):
    counts = Counter(text.lower().split())
    return [counts[word] for word in vocabulary]

def search(query):
    query_vector = encode(query)
    results = []

    for doc in documents:
        doc_vector = encode(doc)

        score = sum(
            q * d
            for q, d in zip(query_vector, doc_vector)
        )

        results.append((doc, score))

    return sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )


for document, score in search("python backend"):
    print(score, document)
