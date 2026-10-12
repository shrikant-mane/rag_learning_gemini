from pathlib import Path


def load_txt(file_path: str) -> str:
    path = Path(file_path)
    print(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    text = load_txt("data/company_policy.txt")

    print("Extracted content:")
    print(text)

    print("\nCharacter count:", len(text))