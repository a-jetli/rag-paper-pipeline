import fitz


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF.

    Opens with fitz.open(), iterates all pages, calls page.get_text().
    Joins pages with double newline.
    Returns empty string (not raises) if PDF is malformed or empty.

    Returns: full text as a single string
    """
    try:
        doc = fitz.open(pdf_path)
        text_parts = []

        for page in doc:
            text = page.get_text()
            text_parts.append(text)

        doc.close()
        return "\n\n".join(text_parts)
    except Exception:
        return ""
