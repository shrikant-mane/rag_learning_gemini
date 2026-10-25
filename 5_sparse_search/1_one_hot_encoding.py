
vocabulary = ["python", "java", "aws"]

def one_hot_encode(word):
    return [1 if token == word else 0 for token in vocabulary]

print(one_hot_encode("python"))
# [1, 0, 0]

print(one_hot_encode("java"))
# [0, 1, 0]

print(one_hot_encode("aws"))
# [0, 0, 1]


def encode_document(text):
    words = set(text.split())
    return [1 if token in words else 0 for token in vocabulary]


doc1 = encode_document("python aws")
doc2 = encode_document("java aws")
query = one_hot_encode("python")

def similarity(query, document):
    return sum(q * d for q, d in zip(query, document))

print(similarity(query, doc1))  # 1
print(similarity(query, doc2))  # 0
