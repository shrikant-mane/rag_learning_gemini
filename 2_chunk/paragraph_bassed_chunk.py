import re

def paragraph_chunking(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


text = """Leave Policy

Employees receive 20 days of annual leave.

Sick Leave

Employees must report sickness according to company policy."""

chunks = paragraph_chunking(text)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:\n{chunk}")