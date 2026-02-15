import os
GEMINI_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PERSIST_DIRECTORY_PATH= os.path.join(PROJECT_ROOT, "chroma_db")
