import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import EMBEDDING_MODEL

_embedder = None


def _get_embedder() -> HuggingFaceEmbeddings:
    """
    Return a shared HuggingFaceEmbeddings instance, creating it on first call.

    Returns
    -------
    HuggingFaceEmbeddings
        A reusable embedding model instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder


def create_vector_store(
    documents: list[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "document_chunks",
) -> Chroma:
    """
    Embed documents and store them in a persistent ChromaDB vector store.

    Deletes any existing store at ``persist_directory`` first so the build
    is always clean.

    Parameters
    ----------
    documents : list[Document]
        A list of LangChain Document objects to embed and store.
    persist_directory : str, optional
        Path on disk where ChromaDB will persist data.
        Default is ``"./chroma_db"``.
    collection_name : str, optional
        Name of the ChromaDB collection. Default is ``"document_chunks"``.

    Returns
    -------
    Chroma
        A LangChain Chroma vector store backed by a PersistentClient.
    """
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    return Chroma.from_documents(
        documents=documents,
        embedding=_get_embedder(),
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def load_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "document_chunks",
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk without re-embedding.

    Parameters
    ----------
    persist_directory : str, optional
        Path on disk where the ChromaDB data was previously persisted.
        Default is ``"./chroma_db"``.
    collection_name : str, optional
        Name of the ChromaDB collection. Default is ``"document_chunks"``.

    Returns
    -------
    Chroma
        A LangChain Chroma vector store loaded from the given directory.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=_get_embedder(),
        collection_name=collection_name,
    )


def query_vector_store(
    vector_store: Chroma,
    query: str,
    k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Query the vector store and return the most relevant chunks with scores.

    Parameters
    ----------
    vector_store : Chroma
        A LangChain Chroma vector store to search.
    query : str
        The search query string.
    k : int, optional
        Number of top results to return. Default is 5.

    Returns
    -------
    list[tuple[Document, float]]
        A list of ``(Document, score)`` tuples. Lower score means a
        better match.
    """
    return vector_store.similarity_search_with_score(query, k=k)


# Test Example
if __name__ == "__main__":
    import glob
    from extraction.pdf_loader import load_and_chunk_pdf

    # 1. Load and chunk a PDF
    pdfs = glob.glob("../sample_docs/*.pdf")
    if not pdfs:
        print("No PDFs found in sample_docs/")
        raise SystemExit(1)

    pdf_path = pdfs[0]
    print(f"Loading PDF: {pdf_path}")
    documents = load_and_chunk_pdf(pdf_path)
    print(f"Total chunks: {len(documents)}")

    # 2. Create the vector store (deletes old one first)
    print("\nCreating vector store (this may take a moment)...")
    vs = create_vector_store(documents)
    print("Vector store created and persisted to ./chroma_db")

    # 3. Run test queries
    test_queries = [
        "as Borrower",
        "Lender",
        "How much money is the CREDIT AGREEMENT worth",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = query_vector_store(vs, query, k=10)

        for i, (doc, score) in enumerate(results, 1):
            page = doc.metadata.get("page", "?")
            char_count = len(doc.page_content)
            print(f"\n  Chunk {i} (page {page}, score: {score:.4f}, chars: {char_count}):")
            print(f"  {doc.page_content[:200]}...")
