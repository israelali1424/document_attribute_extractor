import os
import shutil
import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, CHROMA_DB_PERSIST_DIRECTORY_PATH
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
    collection_name: str = "document_chunks",
) -> chromadb.Collection:
    """
    Embed documents and store them in a persistent ChromaDB collection.

    Deletes any existing store at ``persist_directory`` first so the build
    is always clean.

    Parameters
    ----------
    documents : list[Document]
        A list of LangChain Document objects to embed and store.
        Path on disk where ChromaDB will persist data.
        Default is ``"./chroma_db"``.
    collection_name : str, optional
        Name of the ChromaDB collection. Default is ``"document_chunks"``.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection containing the embedded documents.
    """
    if os.path.exists(CHROMA_DB_PERSIST_DIRECTORY_PATH):
        shutil.rmtree(CHROMA_DB_PERSIST_DIRECTORY_PATH)

    client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_DIRECTORY_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    embedder = _get_embedder()

    ids = []
    texts = []
    metadatas = []

    for i, doc in enumerate(documents):
        ids.append(f"chunk_{i}")
        texts.append(doc.page_content)
        metadatas.append({
            "page": doc.metadata["page"],
            "chunk_index": doc.metadata["chunk_index"],
            "source": doc.metadata["source"],
        })

    embeddings = embedder.embed_documents(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def load_vector_store(
    collection_name: str = "document_chunks",
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk without re-embedding.

    Parameters
    ----------
    collection_name : str, optional
        Name of the ChromaDB collection. Default is ``"document_chunks"``.

    Returns
    -------
    chromadb.Collection
        The existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_DIRECTORY_PATH)
    return client.get_collection(name=collection_name)


def query_vector_store(
    collection: chromadb.Collection,
    query: str,
    k: int = 5,
) -> dict:
    """
    Query the collection and return the most relevant document chunks.

    Parameters
    ----------
    collection : chromadb.Collection
        A ChromaDB collection to search.
    query : str
        The search query string.
    k : int, optional
        Number of top results to return. Default is 5.

    Returns
    -------
    dict
        Raw ChromaDB query results with keys ``documents``, ``metadatas``,
        ``distances``, and ``ids``.
    """
    # Validate that k does not exceed collection size
    collection_size = collection.count()
    if k > collection_size:
        raise ValueError(
            f"Requested k={k} results, but collection only contains {collection_size} documents. "
            f"Set k to a value <= {collection_size}."
        )
    embedder = _get_embedder()
    query_embedding = embedder.embed_query(query)

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )


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
    collection = create_vector_store(documents)
    print(f"Vector store created — {collection.count()} chunks stored in chroma_db")

    # 3. Run test queries
    test_queries = [
       "CREDIT AND SECURITY AGREEMENT"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = query_vector_store(collection, query, k=1111)

        for i in range(len(results["documents"][0])):
            doc_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            char_count = len(doc_text)
            print(f"\n  Chunk {i+1} (page {metadata['page']}, distance: {distance:.4f}, chars: {char_count}):")
            print(f"  {doc_text[:200]}...")
