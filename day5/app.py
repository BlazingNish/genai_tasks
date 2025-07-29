import streamlit as st
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, NLTKTextSplitter
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
from pypdf import PdfReader


@st.cache_resource
def download_nltk_data():
    """Downloads the required NLTK tokenizers if they are not found."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads a SentenceTransformer model and caches it."""
    return SentenceTransformer(model_name)


download_nltk_data()
with st.spinner("Loading embedding model... This may take a moment."):
    model = load_embedding_model()


st.set_page_config(layout="wide", page_title="Text Chunker App")

st.title("LangChain Text Chunker")
st.write("---")
st.markdown("""
    <p style='text-align: justify;'>
    This app demonstrates different text chunking methods from LangChain. Upload a text file, 
    choose a method, adjust the parameters, and see how the document is split. This is a key step 
    in building Retrieval-Augmented Generation (RAG) systems.
    </p>
    """, unsafe_allow_html=True)
st.write("---")

st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

chunker_type = st.sidebar.selectbox(
    "Select Chunker Method",
    ("Recursive", "Fixed-size", "Sentence Splitter")
)

if chunker_type in ["Recursive", "Fixed-size"]:
    chunk_size = st.sidebar.slider("Chunk Size (characters)", 100, 2000, 500, 50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", 0, 500, 50, 10)
else:
    st.sidebar.info("The Sentence Splitter divides text by sentences and does not use Chunk Size or Overlap.")

process_button = st.sidebar.button("Chunk Document", type="primary")

if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if process_button and uploaded_file is not None:

    st.session_state.chunks = []
    st.session_state.embeddings = None
    
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PdfReader(uploaded_file)
        file_contents = ""
        for page in pdf_reader.pages:
            file_contents += page.extract_text() or ""
    else:
        file_contents = uploaded_file.read().decode("utf-8")
    
    splitter = None
    if chunker_type == "Recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunker_type == "Sentence Splitter":
        splitter = NLTKTextSplitter()
    elif chunker_type == "Fixed-size":
        splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    st.session_state.chunks = splitter.split_text(file_contents)
    st.success(f"Document chunked successfully into {len(st.session_state.chunks)} chunks.")

elif process_button and uploaded_file is None:
    st.warning("Please upload a file first.")

if st.session_state.chunks:
    st.subheader(f"Results from '{chunker_type}' Splitter")
    
    max_chunks_to_show = min(5, len(st.session_state.chunks))
    if max_chunks_to_show > 0:
        st.info(f"Showing the first {max_chunks_to_show} of {len(st.session_state.chunks)} chunks below.")
        for i, chunk in enumerate(st.session_state.chunks[:max_chunks_to_show]):
            with st.expander(f"Chunk {i+1} (Length: {len(chunk)} characters)"):
                st.write(chunk)
    else:
        st.warning("No chunks were generated. The document might be empty or parameters too restrictive.")

    st.write("---")

    st.header("Step 2: Generate Embeddings")
    st.info(f"The **'all-MiniLM-L6-v2'** model will be used to create embeddings.")

    embed_button = st.button("Generate Embeddings for Chunks", type="primary")

    if embed_button:
        with st.spinner("Generating embeddings... Please wait."):
            st.session_state.embeddings = model.encode(st.session_state.chunks)
        st.success("Embeddings generated successfully!")

if st.session_state.embeddings is not None:
    st.subheader("Embedding Results")

    col1, col2 = st.columns(2)
    col1.metric(label="Number of Vectors Generated", value=st.session_state.embeddings.shape[0])
    col2.metric(label="Dimensions per Vector", value=st.session_state.embeddings.shape[1])
    
    with st.expander("View a sample embedding vector"):
        st.code(st.session_state.embeddings[0], line_numbers=False)
