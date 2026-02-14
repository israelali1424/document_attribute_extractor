# Document Attribute Extractor

Upload any PDF, define the attributes you want extracted, and the system pulls them out using AI

## Tech Stack

- Python
- LangChain
- Google Gemini
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

### 5. Run the app

```bash
streamlit run app.py
```
