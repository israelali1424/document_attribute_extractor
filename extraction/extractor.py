import json

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from extraction.vector_store import query_vector_store
from config import GEMINI_MODEL, get_secret

SYSTEM_PROMPT = """You are reading excerpts from a legal document.
Based only on these excerpts, extract the value for the requested attribute.
Respond in JSON with exactly these keys:
- "value": the extracted answer (use "not found" if the answer is not in the excerpts)
- "confidence": "high", "medium", or "low" based on how clearly the answer appears
- "reasoning": one sentence explaining where you found it

Respond with ONLY the JSON object, no markdown fences, no extra text."""


def extract_attribute(
    collection: chromadb.Collection,
    attribute_name: str,
    k: int = 5,
) -> dict:
    """
    Query the collection using the attribute name, then ask Gemini
    to extract its value from the retrieved chunks.

    Parameters
    ----------
    collection : chromadb.Collection
        A ChromaDB collection to search.
    attribute_name : str
        The name of the attribute to extract (e.g. "borrower").
    k : int, optional
        Number of top chunks to retrieve. Default is 5.

    Returns
    -------
    dict
        A dict with keys:
        - ``value``: the extracted answer string
        - ``confidence``: one of "high", "medium", or "low"
        - ``source_pages``: list of page numbers the chunks came from
        - ``source_chunks``: list of raw chunk texts used
    """
    api_key = get_secret("GOOGLE_API_KEY")

    results = query_vector_store(collection, attribute_name, k=k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    source_chunks = docs
    source_pages = list(dict.fromkeys(
        meta.get("page") for meta in metas
    ))

    context = "\n\n---\n\n".join(
        f"[Page {metas[i].get('page', '?')}]\n{docs[i]}"
        for i in range(len(docs))
    )

    user_message = f"""Here are the relevant excerpts from the document:

{context}

Extract the value for: {attribute_name}"""

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
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        content = " ".join(parts)
    elif isinstance(content, dict) and "text" in content:
        content = content["text"]
    raw = content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"value": raw, "confidence": "low", "reasoning": "Failed to parse JSON"}

    return {
        "value": parsed.get("value", "not found"),
        "confidence": parsed.get("confidence", "low"),
        "reasoning": parsed.get("reasoning", ""),
        "source_pages": source_pages,
        "source_chunks": source_chunks,
    }


def extract_all_attributes(
    collection,
    attribute_list: list[str],
) -> dict[str, dict]:
    """
    Extract multiple attributes from the document.

    Parameters
    ----------
    collection : chromadb.Collection
        A ChromaDB collection to search.
    attribute_list : list[str]
        A list of attribute names to extract.

    Returns
    -------
    dict[str, dict]
        A mapping of ``{attribute_name: result_dict}`` where each
        ``result_dict`` is the output of ``extract_attribute``.
    """
    results = {}
    for name in attribute_list:
        results[name] = extract_attribute(collection, name)
    return results


def save_results(results: dict, output_path: str = "results.json") -> None:
    """
    Save extraction results to a JSON file.

    Parameters
    ----------
    results : dict
        The output of ``extract_all_attributes``.
    output_path : str, optional
        File path for the output JSON. Default is ``"results.json"``.
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


