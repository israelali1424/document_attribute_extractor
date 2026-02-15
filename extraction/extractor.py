import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from config import GEMINI_MODEL

load_dotenv()

SYSTEM_PROMPT = """You are a document analysis assistant. You will be given chunks of text
from a legal/financial document and a question about a specific attribute.

Instructions:
- Answer ONLY based on the provided chunks. Do not make up information.
- If the answer is not found in the chunks, say "Not found in document."
- Be concise and precise.
- At the end of your answer, on a new line, write your confidence level as exactly
  one of: CONFIDENCE: high, CONFIDENCE: medium, or CONFIDENCE: low."""


def extract_attribute(
    vector_store: Chroma,
    query: str,
    api_key: str = None,
    k: int = 5,
) -> dict:
    """
    Query the vector store for relevant chunks and send them to Gemini
    to extract a specific attribute.

    Parameters
    ----------
    vector_store : Chroma
        A LangChain Chroma vector store to search.
    query : str
        The attribute prompt describing what to extract.
    api_key : str, optional
        Google API key. If not provided, falls back to the GOOGLE_API_KEY
        environment variable.
    k : int, optional
        Number of top chunks to retrieve. Default is 5.

    Returns
    -------
    dict
        A dict with keys:
        - ``answer``: the extracted answer string
        - ``confidence``: one of "high", "medium", or "low"
        - ``source_pages``: list of page numbers the chunks came from
        - ``source_chunks``: list of raw chunk texts used
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")

    results = vector_store.similarity_search_with_score(query, k=k)

    source_chunks = [doc.page_content for doc, _ in results]
    source_pages = list(dict.fromkeys(
        doc.metadata.get("page") for doc, _ in results
    ))

    context = "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc, _ in results
    )

    user_message = f"""Here are the relevant chunks from the document:

{context}

Question: {query}"""

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
    )
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    content = response.content
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    raw_answer = content.strip()

    confidence = "low"
    answer = raw_answer
    for level in ("high", "medium", "low"):
        tag = f"CONFIDENCE: {level}"
        if tag in raw_answer.lower():
            confidence = level
            answer = raw_answer[:raw_answer.lower().rfind("confidence")].strip()
            break

    return {
        "answer": answer,
        "confidence": confidence,
        "source_pages": source_pages,
        "source_chunks": source_chunks,
    }


def extract_all_attributes(
    vector_store: Chroma,
    attributes_dict: dict[str, str],
    api_key: str = None,
) -> dict[str, dict]:
    """
    Extract multiple attributes from the document.

    Parameters
    ----------
    vector_store : Chroma
        A LangChain Chroma vector store to search.
    attributes_dict : dict[str, str]
        A mapping of ``{attribute_name: prompt}`` where each prompt
        describes the attribute to extract.
    api_key : str, optional
        Google API key. If not provided, falls back to the GOOGLE_API_KEY
        environment variable.

    Returns
    -------
    dict[str, dict]
        A mapping of ``{attribute_name: result_dict}`` where each
        ``result_dict`` is the output of ``extract_attribute``.
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    results = {}
    for name, prompt in attributes_dict.items():
        results[name] = extract_attribute(vector_store, prompt, api_key)
    return results


if __name__ == "__main__":
    from extraction.vector_store import load_vector_store

    print("Loading vector store...")
    vs = load_vector_store()

    result = extract_attribute(vs, "Who is the borrower?")

    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Source pages: {result['source_pages']}")
