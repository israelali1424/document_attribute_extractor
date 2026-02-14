import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """
    Extract text from a PDF page by page, skipping empty pages.

    Parameters
    ----------
    file_path : str
        Path to the PDF file to load.

    Returns
    -------
    list[tuple[int, str]]
        A list of tuples where each tuple contains the 1-based page number
        and the extracted text for that page.
    """
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i + 1, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
    source: str = "",
) -> list[Document]:
    """
    Split page tuples into chunked LangChain Document objects with metadata.

    Parameters
    ----------
    pages : list[tuple[int, str]]
        A list of tuples where each tuple contains the 1-based page number
        and the extracted text for that page.
    chunk_size : int
        Maximum number of characters in each chunk produced by the splitter.
    chunk_overlap : int
        Number of overlapping characters between adjacent chunks.
    source : str, optional
        The source filename to store in each Document's metadata.
        Default is an empty string.

    Returns
    -------
    list[Document]
        A list of `langchain_core.documents.Document` objects. Each document's
        `page_content` contains a chunk of text, and `metadata` contains
        `page`, `chunk_index`, and `source` keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = []
    for page_number, text in pages:
        chunks = splitter.split_text(text)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "source": source,
                    },
                )
            )
    return documents


def load_and_chunk_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Extract text from a PDF and split it into chunked Document objects.

    Parameters
    ----------
    file_path : str
        Path to the PDF file to load.
    chunk_size : int, optional
        Maximum number of characters in each chunk produced by the splitter.
        Default is 1000.
    chunk_overlap : int, optional
        Number of overlapping characters between adjacent chunks.
        Default is 200.

    Returns
    -------
    list[Document]
        A list of `langchain_core.documents.Document` objects. Each document's
        `page_content` contains a chunk of text, and `metadata` contains
        `page`, `chunk_index`, and `source` keys.
    """
    pages = extract_text_from_pdf(file_path)
    if not pages:
        return []
    source = os.path.basename(file_path)
    return chunk_pages(pages, chunk_size, chunk_overlap, source=source)
