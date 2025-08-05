import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import nltk
from pypdf import PdfReader
import os
import groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Day 7",
    layout="wide"
)

MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

@st.cache_resource
def load_embedding_model(model_name=MODEL_NAME):
    with st.spinner(f"Loading embedding model '{model_name}'..."):
        return SentenceTransformer(model_name)
    
download_nltk_data()
model = load_embedding_model()

if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'groq_client' not in st.session_state:
    st.session_state.groq_client = None

st.title("Document Q&A with FAISS")
st.markdown("""
This application uses a Retrieval-Augmented Generation (RAG) pipeline to answer your questions about a document.
1.  **Upload** a `.txt` or `.pdf` document.
2.  **Configure** the chunking method and provide your Groq API key.
3.  **Build Index** to process and create a searchable vector store of the document.
4.  **Ask a Question** to get a synthesized answer from an LLM, backed by the most relevant text chunks.
""")
st.write("---")

with st.sidebar:
    st.header("Controls")

    st.subheader("1: Upload Document")
    uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])

    st.subheader("2: Configure & Index")
    chunker_type = st.selectbox(
        "Chunking Method",
        ("Recursive", "Sentence Splitter")
    )

    groq_api_key = ""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.sidebar.success("Groq API key loaded from .env file!", icon="✅")
    except Exception as e:
        st.sidebar.error(f"Error loading Groq API key: {e}", icon="🚨")
    
    process_button = st.button("Build FAISS Index", type="primary", disabled=not uploaded_file)
    
    if process_button:
        if not groq_api_key:
            st.error("Please enter your Groq API key to proceed.")
        else:
            try:
                st.session_state.groq_client = groq.Groq(api_key=groq_api_key)
                st.session_state.groq_client.models.list()
            except Exception as e:
                st.error(
                    f"Failed to initialize Groq client. Please check your API key. Error: {e}")
                st.stop()

            st.session_state.chunks = []
            st.session_state.faiss_index = None
            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PdfReader(uploaded_file)
                file_contents = "".join(
                    page.extract_text() or "" for page in pdf_reader.pages)
            else:
                file_contents = uploaded_file.read().decode("utf-8")

            with st.spinner("Chunking document..."):
                if chunker_type == "Recursive":
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700, chunk_overlap=70)
                    st.session_state.chunks = splitter.split_text(
                        file_contents)
                else:
                    lines = file_contents.splitlines()
                    processed_text = " ".join(line.strip() for line in lines if line.strip())
                    st.session_state.chunks = nltk.sent_tokenize(
                        processed_text)

            if st.session_state.chunks:
                with st.spinner("Generating embeddings and building FAISS index..."):
                    embeddings = model.encode(st.session_state.chunks)
                    d = embeddings.shape[1]
                    index = faiss.IndexFlatL2(d)
                    index.add(np.array(embeddings, dtype='float32'))
                    st.session_state.faiss_index = index
                st.success(
                    f"FAISS index built successfully with {len(st.session_state.chunks)} chunks.")
            else:
                st.error("Could not extract any text from the document.")

    st.subheader("3. Ask a Question")
    query_text = st.text_input(
        "Enter your question:", disabled=not st.session_state.faiss_index)
    search_button = st.button("Search", disabled=not query_text)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Document Chunks")
    if st.session_state.chunks:
        st.info(
            f"Document was split into **{len(st.session_state.chunks)}** chunks using the **'{chunker_type}'** method.")
        with st.expander("View Chunks"):
            for i, chunk in enumerate(st.session_state.chunks):
                st.markdown(f"**Chunk {i+1}:**\n> {chunk}")
    else:
        st.info(
            "Upload a document and click 'Build FAISS Index' to see the chunks here.")

with col2:
    st.subheader("Query Results")
    if search_button and st.session_state.faiss_index:
        with st.spinner("Searching for relevant chunks..."):
            query_embedding = model.encode([query_text])

            k = 5
            distances, indices = st.session_state.faiss_index.search(
                np.array(query_embedding, dtype='float32'), k)

            relevant_chunks = [st.session_state.chunks[i] for i in indices[0]]
            context = "\n\n---\n\n".join(relevant_chunks)

        with st.spinner("Generating answer with Groq..."):
            try:
                system_prompt = """
                You are a helpful AI assistant. You are tasked with answering questions about a document.
                You will be given a user's question and a series of text snippets from the document.
                Your task is to synthesize an answer to the user's question based *only* on the provided text snippets.
                If the answer is not contained within the provided text, state that the information is not available in the document.
                Do not use any external knowledge.
                """

                user_prompt = f"Question: {query_text}\n\nContext from the document:\n{context}"

                chat_completion = st.session_state.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="llama3-8b-8192",
                )

                llm_response = chat_completion.choices[0].message.content


                st.success("**Generated Answer:**")
                st.markdown(llm_response)
                st.write("---")

                with st.expander("View Relevant Source Chunks"):
                    for i, idx in enumerate(indices[0]):
                        st.markdown(
                            f"**Source {i+1} (Score: {distances[0][i]:.2f}):**")
                        st.markdown(f"> {st.session_state.chunks[idx]}")

            except Exception as e:
                st.error(
                    f"An error occurred while communicating with Groq: {e}")

    else:
        st.info(
            "Enter a question in the sidebar and click 'Search' to see the AI-generated answer here.")
