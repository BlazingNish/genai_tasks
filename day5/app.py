import streamlit as st
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, NLTKTextSplitter
import nltk


@st.cache_resource
def download_nltk_data():
    """Downloads the required NLTK tokenizers if they are not found."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

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

uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

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

if process_button and uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    
    splitter = None
    if chunker_type == "Recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunker_type == "Fixed-size":
        splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif chunker_type == "Sentence Splitter":
        splitter = NLTKTextSplitter()

    chunks = splitter.split_text(file_contents)
    
    st.header(f"Results from '{chunker_type}' Splitter")
    st.metric(label="Total Chunks Generated", value=len(chunks))
    st.write("---")
    
    st.info("Showing the first 5 chunks below.")
    for i, chunk in enumerate(chunks[:5]):
        with st.expander(f"Chunk {i+1} (Length: {len(chunk)} characters)"):
            st.write(chunk)
            
elif process_button and uploaded_file is None:
    st.warning("Please upload a .txt file first.")