import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from config import GEMINI_MODEL, EMBEDDING_MODEL

load_dotenv()

# Test 1: Can we call Gemini?
print("Testing Gemini LLM connection...")
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
response = llm.invoke("Say 'Hello, the setup is working!' and nothing else.")
print(f"LLM Response: {response.content}")

# Test 2: Can we generate embeddings?
print("\nTesting HuggingFace Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
result = embeddings.embed_query("This is a test sentence.")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"Embedding dimension: {len(result)}")
print(f"First 5 values: {result[:5]}")

# Test 3: Can we use ChromaDB?
print("\nTesting ChromaDB...")
import chromadb
client = chromadb.Client()
collection = client.create_collection("test")
collection.add(documents=["test document"], ids=["1"])
results = collection.query(query_texts=["test"], n_results=1)
print(f"ChromaDB query result: {results['documents']}")

# Test 4: Can we read a PDF?
print("\nTesting pdfplumber...")
import pdfplumber
pdfs = glob.glob("sample_docs/*.pdf")
if pdfs:
    with pdfplumber.open(pdfs[0]) as pdf:
        print(f"PDF loaded: {pdfs[0]}")
        print(f"Number of pages: {len(pdf.pages)}")
        first_page_text = pdf.pages[0].extract_text()
        print(f"First 200 chars: {first_page_text[:200] if first_page_text else 'No text extracted'}")
else:
    print("No PDFs found in sample_docs/ — add some before building the project")

print("\n=== ALL TESTS PASSED — YOU'RE READY TO BUILD ===")
