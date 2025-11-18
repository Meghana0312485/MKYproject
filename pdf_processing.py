import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts full text from a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text("text")  # ensure plain text extraction

        doc.close()

        return text.strip()

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def chunk_text(text, chunk_size=500):
    """
    Splits extracted text into chunks for embeddings.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
