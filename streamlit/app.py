import sys
import os
import json
import tempfile

# Add project root to path so imports work when running from streamlit/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from extraction.pdf_loader import load_and_chunk_pdf
from extraction.vector_store import create_vector_store, load_vector_store
from extraction.extractor import extract_attribute, save_results

st.set_page_config(page_title="Document Attribute Extractor", layout="wide")

CONFIDENCE_COLORS = {
    "high": "#28a745",
    "medium": "#ffc107",
    "low": "#dc3545",
}


# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.header("1. Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

st.sidebar.header("2. Define Attributes")
input_method = st.sidebar.radio("Input method", ["Manual entry", "JSON upload"])

attribute_list = []

if input_method == "Manual entry":
    attributes_text = st.sidebar.text_area(
        "Enter attribute names (one per line)",
        placeholder="borrower\nadministrative agent\ntotal facility amount\nclosing date",
        height=150,
    )
    if attributes_text.strip():
        attribute_list = [
            line.strip() for line in attributes_text.splitlines() if line.strip()
        ]
else:
    uploaded_json = st.sidebar.file_uploader("Upload a JSON list of attributes", type=["json"])
    if uploaded_json:
        try:
            attribute_list = json.load(uploaded_json)
            if not isinstance(attribute_list, list):
                st.sidebar.error("JSON must be a list of strings.")
                attribute_list = []
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file.")

if attribute_list:
    st.sidebar.markdown(f"**{len(attribute_list)} attributes loaded:**")
    for attr in attribute_list:
        st.sidebar.markdown(f"- {attr}")

extract_clicked = st.sidebar.button(
    "Extract",
    disabled=not (uploaded_pdf and attribute_list),
    use_container_width=True,
)


# ── Main Area ───────────────────────────────────────────────────────

st.title("Document Attribute Extractor")
st.markdown("Upload a PDF, define attributes, and extract structured data using AI.")

if extract_clicked and uploaded_pdf and attribute_list:
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
            collection = create_vector_store(documents)
            st.write(f"Vector store ready ({collection.count()} chunks embedded).")
            status.update(label="Document processed.", state="complete")

        # Step 4: Extract attributes
        results = {}
        progress = st.progress(0, text="Extracting attributes...")

        for i, attr_name in enumerate(attribute_list):
            progress.progress(
                (i) / len(attribute_list),
                text=f"Extracting: {attr_name}...",
            )
            results[attr_name] = extract_attribute(collection, attr_name)

        progress.progress(1.0, text="Extraction complete.")

        # Step 5: Display results
        st.header("Results")

        for attr_name, result in results.items():
            value = result.get("value", "not found")
            confidence = result.get("confidence", "low").lower()
            reasoning = result.get("reasoning", "")
            source_pages = result.get("source_pages", [])
            source_chunks = result.get("source_chunks", [])

            color = CONFIDENCE_COLORS.get(confidence, "#6c757d")

            st.subheader(attr_name.title())

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
                        key=f"{attr_name}_chunk_{j}",
                    )

            st.divider()

        # Step 6: Save and offer download
        results_json = json.dumps(results, indent=2)
        save_results(results)

        st.download_button(
            label="Download results as JSON",
            data=results_json,
            file_name="results.json",
            mime="application/json",
        )

    finally:
        os.unlink(tmp_path)

elif not extract_clicked:
    st.info("Upload a PDF and define attributes in the sidebar, then click Extract.")
