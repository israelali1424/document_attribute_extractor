# Document Attribute Extractor

Upload any PDF, define the attributes you want extracted, and the system pulls them out using AI

## Tech Stack

- Python
- LangChain
- Google Gemini (LLM)
- HuggingFace Sentence Transformers (embeddings)
- ChromaDB
- pdfplumber
- Streamlit

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/israelali1424/document_attribute_extractor.git
cd document_attribute_extractor
```

### 2. Create a conda environment

```bash
conda create -n document-attribute_-extractor python=3.11
conda document-attribute-extractor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Verify your setup

Run the test script to confirm that your API key, embeddings, ChromaDB, and PDF reading are all working correctly:

```bash
python test_setup.py
```

If everything is configured properly, you should see `ALL TESTS PASSED` at the end.

### 6. Run the app

```bash
streamlit run app.py
```

## Note

The `chroma_db/` folder stores embeddings on disk so the app doesn't re-embed every time it runs. If you change the chunking logic or text cleaning, delete the `chroma_db/` folder and re-run to regenerate fresh embeddings.
