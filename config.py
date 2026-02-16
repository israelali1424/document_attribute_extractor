import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PERSIST_DIRECTORY_PATH = os.path.join(PROJECT_ROOT, "chroma_db")


def get_secret(key: str) -> str:
    """
    Retrieve a secret by key.

    Checks ``st.secrets`` first (Streamlit Cloud), then falls back to
    environment variables (local ``.env`` loaded via python-dotenv).

    Parameters
    ----------
    key : str
        The secret/environment variable name.

    Returns
    -------
    str
        The secret value.

    Raises
    ------
    KeyError
        If the key is not found in either source.
    """
    try:
        import streamlit as st
        return st.secrets[key]
    except (ImportError, KeyError, FileNotFoundError):
        value = os.getenv(key)
        if value is None:
            raise KeyError(f"Secret '{key}' not found in st.secrets or environment variables.")
        return value
