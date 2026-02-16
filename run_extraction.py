import sys
from extraction.vector_store import load_vector_store
from extraction.extractor import extract_all_attributes, save_results
from extraction.vector_store import query_vector_store
DEFAULT_ATTRIBUTES = [
    "Who is the borrower in this credit agreement?",
]

# Test Example
def main():
    attributes = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ATTRIBUTES

    print("Loading vector store...")
    vs = load_vector_store()
    print(f"Vector store created — {vs.count()} chunks stored in ./chroma_db")
    print(f"Extracting {len(attributes)} attributes...")


    results = extract_all_attributes(vs, attributes)

    for name, result in results.items():
        print(f"\n{'='*60}")
        print(f"Attribute: {name}")
        print("=" * 60)
        print(f"  Value: {result['value']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Source Chunks: {result['source_chunks']}")
        print(f"  Source pages: {result['source_pages']}")

    save_results(results)
    print(f"\nResults saved to results.json")


if __name__ == "__main__":
    main()
