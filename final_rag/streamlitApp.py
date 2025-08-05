import streamlit as st
import weaviate
from sentence_transformers import SentenceTransformer
import os
from groq import Groq
from bert_score import score as bert_scorer
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "PDFChunksBigger"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'

@st.cache_resource
def get_components():
    """Connects to Weaviate, loads models, and initializes clients."""
    client = weaviate.connect_to_local(skip_init_checks=True)
    model = SentenceTransformer(EMBEDDING_MODEL)
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error(
            "Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        st.stop()
    return client, model, groq_client

def run_rag_pipeline(question, collection, embedding_model, groq_client):
    """Runs the full RAG pipeline and returns the answer, context, and sources."""
    query_vector = embedding_model.encode(question).tolist()
    response = collection.query.near_vector(
        near_vector=query_vector, limit=5,
        # Make sure to retrieve all needed properties
        return_properties=["text", "filename", "page_number"]
    )

    if not response.objects:
        return "No relevant information found.", "", []

    context = "\n\n---\n\n".join([obj.properties['text']
                                 for obj in response.objects])
    system_prompt = f"Use the following context to answer the user's question.\n\nCONTEXT:\n{context}"

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        model="llama3-8b-8192", temperature=0.2,
    )
    return chat_completion.choices[0].message.content, context, response.objects


def llm_as_a_judge(question, context, answer, groq_client):
    """Uses a powerful LLM to evaluate the RAG output."""
    judge_prompt = f"""
    You are an impartial AI evaluator. Your task is to assess the quality of a generated answer based on a provided context and a user's question.
    Provide your evaluation in a JSON format with scores from 1 to 5. Do not add any text before or after the JSON object.

    **Criteria:**
    1.  **Faithfulness (1-5):** How well is the answer supported by the context? A score of 5 means fully supported. A score of 1 means not supported at all.
    2.  **Answer Relevance (1-5):** How well does the answer address the user's question? A score of 5 means it's a direct and complete answer.

    **[CONTEXT]:**
    {context}
    **[USER'S QUESTION]:**
    {question}
    **[GENERATED ANSWER]:**
    {answer}

    **Your JSON Evaluation:**
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            model="llama3-70b-8192", temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        return {"faithfulness_score": 0, "answer_relevance_score": 0}



st.set_page_config(layout="wide")
st.title("RAG Chatbot")


try:
    weaviate_client, embedding_model, groq_client = get_components()
    chunks_collection = weaviate_client.collections.get(COLLECTION_NAME)
except Exception as e:
    st.error(
        f"Could not initialize components. Please check your setup. Error: {e}")
    st.stop()

with st.sidebar:
    st.header("RAG Pipeline Evaluation 📊")
    st.markdown("---")

    questions_data = """What data sources were utilized in this study to examine millennial mobile payment users?
How did the researchers define a "mobile payment user" versus a "non-user" for their analysis?
What was the most common technical root cause for ransomware attacks in 2025, and what percentage of incidents did it account for?
What percentage of organizations that had their data encrypted were able to recover it, and what was the most common recovery method?
What was the value of the Index of Consumer Sentiment in June 2025, and what was the trend compared to the previous month?
"""

    ground_truth_data = """The study used data from two primary sources to analyze the financial capability of American millennials. The first was the 2015 National Financial Capability Study (NFCS), a triennial survey that assesses financial capability among American adults. The second was the 2016 GFLEC Mobile Payment Survey, which was designed and fielded by the researchers to gather additional information specifically on mobile payment users.
The definition was based on the response to the question: "How often do you use your mobile phone to pay for a product or service in person at a store, gas station, or restaurant...?". Users were defined as respondents who answered "frequently" or "sometimes". Non-users were defined as respondents who answered "never". Individuals who responded with "don't know" or "prefer not to say" were removed from the sample for the analysis.
For the third consecutive year, exploited vulnerabilities was the most common technical root cause of ransomware attacks. This method was used to penetrate organizations in 32% of all reported incidents
In 2025, 97% of organizations that experienced data encryption were able to recover it. The most frequent recovery method was using backups, which was employed in 54% of incidents where data was restored.
The Index of Consumer Sentiment was 60.7 in June 2025. This represented a 16% surge from May, marking the first increase in the index in six months.
"""

    with st.expander("1. LLM-as-a-Judge Evaluation"):
        st.markdown(
            "Evaluates **faithfulness** and **relevance** using Llama3-70b.")
        judge_questions = st.text_area(
            "Questions for Judge (one per line)", value=questions_data, height=150)
        if st.button("Run LLM-as-a-Judge"):
            question_list = [q.strip()
                             for q in judge_questions.split('\n') if q.strip()]
            if not question_list:
                st.warning("Please provide at least one question.")
            else:
                with st.spinner(f"Running LLM-as-a-Judge on {len(question_list)} questions..."):
                    results = []
                    for q in question_list:
                        answer, context, _ = run_rag_pipeline(
                            q, chunks_collection, embedding_model, groq_client)
                        if context:
                            scores = llm_as_a_judge(
                                q, context, answer, groq_client)
                            results.append({"Question": q, "Answer": answer, "Faithfulness": scores.get(
                                "Faithfulness", 0), "Relevance": scores.get("Answer Relevance", 0)})
                    st.session_state.judge_results = pd.DataFrame(results)

    with st.expander("2. Ground Truth-Based Evaluation (BERTScore)"):
        st.markdown(
            "Evaluates generated answers against your provided correct answers.")
        gt_questions = st.text_area(
            "Questions (one per line)", value=questions_data, height=150, key="gt_q")
        ground_truth_answers = st.text_area(
            "Ground Truth Answers (one per line)", value=ground_truth_data, height=150, key="gt_a")
        if st.button("Run BERTScore F1 Evaluation"):
            question_list = [q.strip()
                             for q in gt_questions.split('\n') if q.strip()]
            answer_list = [a.strip()
                           for a in ground_truth_answers.split('\n') if a.strip()]
            if len(question_list) == len(answer_list) and question_list:
                with st.spinner("Calculating BERTScore..."):
                    gen_answers = [run_rag_pipeline(q, chunks_collection, embedding_model, groq_client)[
                        0] for q in question_list]
                    P, R, F1 = bert_scorer(
                        gen_answers, answer_list, lang="en", model_type='roberta-large')
                    st.session_state.bert_results = {"dataframe": pd.DataFrame({"Question": question_list, "Generated Answer": gen_answers, "F1 Score": F1.tolist(
                    ), "Precision": P.tolist(), "Recall": R.tolist()}), "avg_f1": F1.mean().item(), "avg_precision": P.mean().item(), "avg_recall": R.mean().item()}

    if 'judge_results' in st.session_state:
        st.subheader("LLM-as-a-Judge Results")
        df = st.session_state.judge_results
        st.metric("Avg. Faithfulness", f"{df['Faithfulness'].mean():.2f}/5")
        st.metric("Avg. Relevance", f"{df['Relevance'].mean():.2f}/5")
        st.dataframe(df)

    if 'bert_results' in st.session_state:
        st.subheader("BERTScore F1 Results")
        st.metric("Average F1 Score",
                  f"{st.session_state.bert_results['avg_f1']:.4f}")
        st.metric("Average Precision",
                  f"{st.session_state.bert_results['avg_precision']:.4f}")
        st.metric("Average Recall",
                  f"{st.session_state.bert_results['avg_recall']:.4f}")
        st.dataframe(st.session_state.bert_results['dataframe'])

st.header("Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Show Retrieved Sources"):
                for source in message["sources"]:
                    st.write(source)

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, _, sources_objects = run_rag_pipeline(
                prompt, chunks_collection, embedding_model, groq_client)

            st.markdown(response_text)

            formatted_sources = []
            if sources_objects:
                with st.expander("Show Retrieved Sources"):
                    for i, obj in enumerate(sources_objects):
                        properties = obj.properties
                        source_text = f"**Source {i+1}:** {properties['filename']} (Page: {properties['page_number']})\n> {properties['text']}"
                        st.write(source_text)
                        formatted_sources.append(source_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": formatted_sources
    })
