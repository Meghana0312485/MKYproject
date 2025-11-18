import streamlit as st
import tempfile
from pdf_processing import extract_text_from_pdf, chunk_text
from embeddings import SemanticSearchEngine
from llm_answer import generate_answer

st.set_page_config(page_title="StudyMate AI", page_icon="📘", layout="wide")

st.title("📘 StudyMate – AI Academic Assistant")
st.write("Upload PDFs and ask questions. StudyMate will answer based on your study material.")

uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# Initialize vector engine only once
if "vector_engine" not in st.session_state:
    st.session_state.vector_engine = None

# Process uploaded PDFs
if uploaded_files:
    st.success("PDFs uploaded successfully!")

    all_chunks = []

    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded.read())
            text = extract_text_from_pdf(temp_pdf.name)

            if text.strip():  # avoid empty PDFs
                chunks = chunk_text(text)
                all_chunks.extend(chunks)

    # Create semantic search index
    if all_chunks:
        st.session_state.vector_engine = SemanticSearchEngine()
        st.session_state.vector_engine.create_index(all_chunks)
        st.info("Text extracted & semantic index created!")
    else:
        st.error("No readable text found in the uploaded PDFs.")

# Question input
question = st.text_input("Ask a question about your study materials:")

# Button action
if st.button("Get Answer"):
    if st.session_state.vector_engine is None:
        st.error("Please upload PDFs first.")
    else:
        with st.spinner("Retrieving best context..."):
            retrieved = st.session_state.vector_engine.search(question, top_k=3)

        with st.spinner("Generating answer using IBM Granite LLM..."):
            answer = generate_answer(question, retrieved)

        st.subheader("📘 Answer")
        st.write(answer)

        st.subheader("🔍 Retrieved Context")
        st.write(retrieved)
