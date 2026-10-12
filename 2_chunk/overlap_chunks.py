def chunk_with_overlap(
    text: str,
    chunk_size: int = 50,
    overlap: int = 10,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


text = (
    "Python is useful for AI. "
    "RAG retrieves relevant information. "
    "Embeddings represent text as vectors."
)

chunks = chunk_with_overlap(
    text,
    chunk_size=40,
    overlap=10,
)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk!r}")