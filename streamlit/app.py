import sys
import os
import json
import tempfile

# Add project root to path so imports work when running from streamlit/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from extraction.pdf_loader import load_and_chunk_pdf
from extraction.vector_store import create_vector_store
from extraction.extractor import extract_attribute, save_results

st.set_page_config(page_title="Document Attribute Extractor", layout="wide")

CONFIDENCE_COLORS = {
    "high": "#28a745",
    "medium": "#ffc107",
    "low": "#dc3545",
}

TERM_TO_QUERY = {
    "credit_agreement": {
        "borrower": "Who is the borrower in this credit agreement? ",
        "lender": "Who is the administrative agent and/or lead lender?",
        "credit agreement amount": "How much is the credit agreement",
    },
}


def resolve_query(term: str, doc_type: str) -> str:
    """Map a user search term to a well-formed question, or fall back to raw input."""
    mapping = TERM_TO_QUERY.get(doc_type, {})
    return mapping.get(term.lower().strip(), term)


# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.header("1. Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Document type (default to credit_agreement for now)
doc_type = "credit_agreement"
st.sidebar.header("2. Document Type")
st.sidebar.markdown(f"**Selected:** `{doc_type}`")

# Show available search terms
available_terms = list(TERM_TO_QUERY.get(doc_type, {}).keys())
with st.sidebar.expander("Available search terms", expanded=False):
    for term in available_terms:
        st.markdown(f"- `{term}`")

st.sidebar.header("3. Search")
search_term = st.sidebar.text_input(
    "Enter a search term or question",
    placeholder="e.g. borrower, interest rate, collateral",
)

extract_clicked = st.sidebar.button(
    "Extract",
    disabled=not (uploaded_pdf and search_term.strip()),
    use_container_width=True,
)


# ── Main Area ───────────────────────────────────────────────────────

st.title("Document Attribute Extractor")
st.markdown("Upload a PDF, enter a search term, and extract structured data using AI.")

if extract_clicked and uploaded_pdf and search_term.strip():
    term = search_term.strip()
    query = resolve_query(term, doc_type)
    matched = query != term

    if not matched:
        st.warning(f"No mapping found for \"{term}\" — using raw input as query.")

    # Step 1: Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    try:
        # Step 2: Load and chunk the PDF
        with st.status("Processing document...", expanded=True) as status:
            st.write("Extracting text and chunking PDF...")
            documents = load_and_chunk_pdf(tmp_path)
            st.write(f"Created {len(documents)} chunks.")

            # Step 3: Build vector store
            st.write("Building vector store...")
            collection = create_vector_store(documents, in_memory=True)
            st.write(f"Vector store ready ({collection.count()} chunks embedded).")
            status.update(label="Document processed.", state="complete")

        # Step 4: Extract attribute
        with st.spinner(f"Extracting: {term}..."):
            result = extract_attribute(collection, query)

        # Step 5: Display result
        st.header("Result")

        value = result.get("value", "not found")
        confidence = result.get("confidence", "low").lower()
        reasoning = result.get("reasoning", "")
        source_pages = result.get("source_pages", [])
        source_chunks = result.get("source_chunks", [])

        color = CONFIDENCE_COLORS.get(confidence, "#6c757d")

        st.subheader(term.title())

        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            st.markdown(f"**Value:** {value}")
        with col2:
            st.markdown(
                f'<span style="background-color:{color};color:white;'
                f'padding:4px 12px;border-radius:12px;font-size:0.85em;">'
                f'{confidence.upper()}</span>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(f"**Pages:** {', '.join(str(p) for p in source_pages)}")

        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")

        with st.expander("Source chunks"):
            for j, chunk in enumerate(source_chunks):
                st.text_area(
                    f"Chunk {j + 1}",
                    value=chunk,
                    height=120,
                    disabled=True,
                    key=f"{term}_chunk_{j}",
                )

        st.divider()

        # Step 6: Save and offer download
        results = {term: result}
        results_json = json.dumps(results, indent=2)
        save_results(results)

        st.download_button(
            label="Download result as JSON",
            data=results_json,
            file_name="results.json",
            mime="application/json",
        )

    finally:
        os.unlink(tmp_path)

elif not extract_clicked:
    st.info("Upload a PDF and enter a search term in the sidebar, then click Extract.")
