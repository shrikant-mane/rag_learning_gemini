import re

def sentence_chunking(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


text = (
    "Python is a programming language. "
    "RAG retrieves relevant information. "
    "Embeddings represent text as vectors!"
)

for chunk in sentence_chunking(text):
    print(chunk)