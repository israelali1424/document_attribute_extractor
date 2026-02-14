import chromadb
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

COLLECTION_NAME = "document_chunks"

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
) -> chromadb.Collection:
    """
    Embed documents and store them in a persistent ChromaDB collection.

    Parameters
    ----------
    documents : list[Document]
        A list of LangChain Document objects to embed and store.
    persist_directory : str, optional
        Path on disk where ChromaDB will persist data.
        Default is ``"./chroma_db"``.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection containing the embedded documents.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedder = _get_embedder()

    ids = []
    texts = []
    metadatas = []
    embeddings = []

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
    persist_directory: str = "./chroma_db",
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk without re-embedding.

    Parameters
    ----------
    persist_directory : str, optional
        Path on disk where the ChromaDB data was previously persisted.
        Default is ``"./chroma_db"``.

    Returns
    -------
    chromadb.Collection
        The existing ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_collection(name=COLLECTION_NAME)


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

    # 2. Create the vector store
    print("\nCreating vector store (this may take a moment)...")
    collection = create_vector_store(documents)
    print(f"Vector store created — {collection.count()} chunks stored in ./chroma_db")

    # 3. Run test queries
    test_queries = [
        "Who is the borrower?",
        "What is the interest rate?",
        "What triggers a change of control?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        results = query_vector_store(collection, query, k=3)

        for i in range(len(results["documents"][0])):
            doc_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            print(f"\n  Chunk {i+1} (page {metadata['page']}, distance: {distance:.4f}):")
            print(f"  {doc_text[:200]}...")
