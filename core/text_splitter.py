from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits a large text into smaller, manageable chunks.

    Args:
        text: The full text content of the document.

    Returns:
        A list of text chunks.
    """
    # Initialize the text splitter from LangChain.
    # This splitter is designed to be "semantically aware" by trying to split
    # on paragraphs, then sentences, etc., to keep related text together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # The maximum size of each chunk (in characters)
        chunk_overlap=200,    # The number of characters to overlap between chunks
        length_function=len   # The function used to measure chunk size
    )

    # Perform the split
    chunks = text_splitter.split_text(text)
    
    print(f"[INFO] ==> Step 2b: Document successfully split into {len(chunks)} processable chunks.")
    
    return chunks
