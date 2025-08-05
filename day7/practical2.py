
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import nltk
from pypdf import PdfReader
from bert_score import score as bert_score_func
import torch
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()


st.set_page_config(
    page_title="RAG Prompting Technique Comparison",
    page_icon="⚖️",
    layout="wide"
)


MODEL_NAME = 'all-MiniLM-L6-v2'




@st.cache_resource
def download_nltk_data():
    """Downloads the required NLTK 'punkt' tokenizer if not found."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)


@st.cache_resource
def load_embedding_model(model_name=MODEL_NAME):
    """Loads a SentenceTransformer model and caches it."""
    with st.spinner(f"Loading embedding model '{model_name}'..."):
        return SentenceTransformer(model_name)


download_nltk_data()
model = load_embedding_model()


st.session_state.setdefault('chunks', [])
st.session_state.setdefault('faiss_index', None)
st.session_state.setdefault('comparison_results', {})
st.session_state.setdefault('retrieved_context', None)


st.title("RAG Prompting Technique Comparison")
st.markdown("""
This application directly compares the performance of **Role-Based** and **ReAct-Based** prompting in a RAG pipeline.
1.  **Upload & Index:** Prepare your document.
2.  **Define Prompts & Query:** Set up the parameters for the comparison.
3.  **Generate & Compare:** Run both pipelines and see the side-by-side results and BERTScore evaluations.
""")
st.write("---")


with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Groq API Key")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        st.warning("Please set your Groq API key in the `.env` file as `GROQ_API_KEY`.")

    st.subheader("Step 1: Upload & Index")
    uploaded_file = st.file_uploader(
        "Choose a .txt or .pdf file", type=["txt", "pdf"])
    process_button = st.button(
        "Build FAISS Index", type="primary", disabled=not uploaded_file)

    if process_button:
        with st.spinner("Processing document..."):
            st.session_state.chunks, st.session_state.faiss_index, st.session_state.comparison_results = [], None, {}
            if uploaded_file.name.endswith('.pdf'):
                pdf_reader = PdfReader(uploaded_file)
                file_contents = "".join(
                    page.extract_text() or "" for page in pdf_reader.pages)
            else:
                file_contents = uploaded_file.read().decode("utf-8")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=100)
            st.session_state.chunks = splitter.split_text(file_contents)

            if st.session_state.chunks:
                embeddings = model.encode(st.session_state.chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings, dtype='float32'))
                st.session_state.faiss_index = index
                st.success(
                    f"FAISS index built with {len(st.session_state.chunks)} chunks.")
            else:
                st.error("Could not extract text.")

    st.subheader("Step 2: Define Prompts & Query")
    role_prompt_input = st.text_input(
        "Role for Role-Based Prompt:", "You are a literary expert specializing in narrative analysis.")
    query_text = st.text_input(
        "Your Question:", disabled=not st.session_state.faiss_index)
    ground_truth_answer = st.text_area("Ground Truth Answer (for evaluation):",
                                       help="Provide a perfect, human-written answer for BERTScore comparison.",
                                       disabled=not st.session_state.faiss_index)

    compare_button = st.button("Generate & Compare Techniques", disabled=not (
        query_text and ground_truth_answer and groq_api_key))

if compare_button:
    try:
        client = Groq(api_key=groq_api_key)
        st.session_state.comparison_results = {}

        with st.spinner("Retrieving relevant context..."):
            query_embedding = model.encode([query_text])
            k = 4
            _, indices = st.session_state.faiss_index.search(
                np.array(query_embedding, dtype='float32'), k)
            retrieved_chunks = [st.session_state.chunks[i] for i in indices[0]]
            st.session_state.retrieved_context = "\n\n---\n\n".join(
                retrieved_chunks)

        prompt_techniques = {
            "Role-Based": f"System Role: {role_prompt_input}\n\nContext:\n{st.session_state.retrieved_context}\n\nQuestion: {query_text}\n\nBased on the context and your role, provide a direct answer.",
            "ReAct-Based": f"You are a ReAct agent. Your task is to answer the following question based on the provided context. Follow the ReAct framework precisely.\n\nContext:\n{st.session_state.retrieved_context}\n\nQuestion: {query_text}\n\nThought: (Begin by thinking step-by-step about how to answer the question using only the provided context. Identify the key pieces of information needed.)\nAction: (State that you will now formulate the final answer based on your thoughts.)\nAnswer: (Provide the final, concise answer to the question.)"
        }

        for tech, prompt in prompt_techniques.items():
            with st.spinner(f"Generating answer for {tech} prompt..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                )
                generated_answer = chat_completion.choices[0].message.content

            with st.spinner(f"Calculating BERTScore for {tech}..."):
                P, R, F1 = bert_score_func(
                    [generated_answer], [ground_truth_answer], lang="en", verbose=False)
                bert_scores = {"precision": P.mean().item(
                ), "recall": R.mean().item(), "f1": F1.mean().item()}

            st.session_state.comparison_results[tech] = {
                "answer": generated_answer,
                "prompt": prompt,
                "scores": bert_scores
            }
        st.success("Comparison complete!")
    except Exception as e:
        st.error(f"An error occurred: {e}")


st.header("📊 Comparison Results")

if not st.session_state.faiss_index:
    st.info("Upload a document and build the FAISS index to begin.")
elif not st.session_state.comparison_results:
    st.info("Define your prompts and question in the sidebar, then click 'Generate & Compare Techniques'.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Role-Based Prompting")
        role_results = st.session_state.comparison_results.get(
            "Role-Based", {})
        if role_results:
            st.write(role_results.get("answer", "No answer generated."))
            scores = role_results.get("scores", {})
            st.metric("BERTScore F1", f"{scores.get('f1', 0):.4f}")

    with col2:
        st.subheader("ReAct-Based Prompting")
        react_results = st.session_state.comparison_results.get(
            "ReAct-Based", {})
        if react_results:
            st.write(react_results.get("answer", "No answer generated."))
            scores = react_results.get("scores", {})
            st.metric("BERTScore F1", f"{scores.get('f1', 0):.4f}")

    st.write("---")
    st.subheader("🏆 Verdict")
    role_f1 = st.session_state.comparison_results.get(
        "Role-Based", {}).get("scores", {}).get("f1", 0)
    react_f1 = st.session_state.comparison_results.get(
        "ReAct-Based", {}).get("scores", {}).get("f1", 0)

    if role_f1 > react_f1:
        st.success(
            f"**Role-Based Prompting performed better** with an F1 score of {role_f1:.4f} vs {react_f1:.4f}.")
    elif react_f1 > role_f1:
        st.success(
            f"**ReAct-Based Prompting performed better** with an F1 score of {react_f1:.4f} vs {role_f1:.4f}.")
    else:
        st.info("Both techniques performed equally.")

    with st.expander("View Details (Retrieved Context and Prompts)"):
        st.text_area("Retrieved Context (Same for Both):",
                     st.session_state.retrieved_context, height=200)
        st.text_area("Role-Based Prompt:", st.session_state.comparison_results.get(
            "Role-Based", {}).get("prompt", ""), height=200)
        st.text_area("ReAct-Based Prompt:", st.session_state.comparison_results.get(
            "ReAct-Based", {}).get("prompt", ""), height=200)
