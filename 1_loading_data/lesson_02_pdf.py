from pathlib import Path
import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    pages = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text"),
            })

    return pages


if __name__ == "__main__":
    pages = load_pdf("data/rag_learning_sample.pdf")

    for page in pages:
        print(f"Page {page['page_number']}")
        print(page["text"])