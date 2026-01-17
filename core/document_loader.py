from docx import Document

def load_text_from_docx(file_path: str) -> str:
    """
    Loads all text content from a .docx file into a single string.

    Args:
        file_path: The full path to the .docx file.

    Returns:
        A single string containing all the text from the document.
    """
    try:
        doc = Document(file_path)
        # Use a generator expression within join for memory efficiency
        full_text = "\n".join(para.text for para in doc.paragraphs)
        print(f"[INFO] ==> Step 2a: Document loading successful. Extracted text from '{file_path}'.")
        return full_text
    except Exception as e:
        print(f"[ERROR] Failed to load document {file_path}. Error: {e}")
        # Propagate the error to be handled by the API layer
        raise
