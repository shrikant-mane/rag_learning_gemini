
from collections import Counter

documents = [
    "python is easy python",
    "java is easy",
    "python and java"
]

# Step 1: Tokenization
tokenized_docs = [doc.lower().split() for doc in documents]

# Step 2: Build vocabulary
vocabulary = sorted(set(
    word
    for doc in tokenized_docs
    for word in doc
))

print("Vocabulary:", vocabulary)

# Step 3: Convert documents to vectors
def bag_of_words(tokens, vocabulary):
    counts = Counter(tokens)
    return [counts.get(word, 0) for word in vocabulary]

# Step 4: Generate BoW vectors
for doc in tokenized_docs:
    print(bag_of_words(doc, vocabulary))
