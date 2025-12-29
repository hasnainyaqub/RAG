import json
import streamlit as st
import os
import pickle
import faiss
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# ----------------------------
# Load FAISS + metadata
# ----------------------------
@st.cache_resource
def load_vectorstore():
    index = faiss.read_index("faiss_new.index")
    with open("metadata_new.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


index, metadata = load_vectorstore()


@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}
    )


embedder = load_embedder()


# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query, k=5):
    q_emb = embedder.embed_query(query)
    q_emb = np.array([q_emb], dtype="float32")

    _, indices = index.search(q_emb, k)

    return [
        metadata["docstore"][metadata["index_to_docstore_id"][int(i)]]
        for i in indices[0]
    ]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="QUEST RAG App", layout="wide")

st.title("📄 QUEST RAG Assistant")

st.sidebar.header("Model Settings")

provider = st.sidebar.selectbox(
    "Select Provider",
    ["Groq", "OpenRouter"]
)

model_name = None
api_key = None

if provider == "OpenRouter":
    model_name = st.sidebar.selectbox(
        "Select Model",
        [
            "deepseek/deepseek-r1",
            "xwin-13b-instruct",
            "gpt-neox-20b-instruct"
        ]
    )
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

elif provider == "Groq":
    model_name = st.sidebar.selectbox(
        "Select Model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-safeguard-20b",
            "openai/gpt-oss-20b",
            "groq/compound",
            "groq/compound-mini",
            "llama-3.3-70b-versatile"
        ]
    )

    api_key = st.secrets["GROQ_API_KEY"]

    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json"
    }


st.sidebar.markdown("### API Sources")

st.sidebar.markdown(
    "Recommended to use **OpenRouter API** with **deepseek/deepseek-r1** for best results. \n"
    "- **OpenRouter API**  \n"
    "https://openrouter.ai/settings/keys"
)

st.sidebar.markdown(
    "Recommended to use **Groq API** with **gpt-oss-120b** for best results. \n"
    "- **Groq API**  \n"
    "https://console.groq.com/keys"
)


# ----------------------------
# Load LLM dynamically
# ----------------------------
def load_llm(provider, model_name, api_key):

    if provider == "Groq":
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(
            model=model_name,
            temperature=0
        )

    if provider == "OpenRouter":
        os.environ["OPENROUTER_API_KEY"] = api_key
        return ChatOpenAI(
            model="deepseek/deepseek-r1",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0
        )


# ----------------------------
# RAG Answer
# ----------------------------
def rag_answer(query, llm):
    docs = retrieve(query)

    if not docs:
        return "Not found in documents."

    context = "\n\n".join(
        [
            f"Source: {d.metadata.get('source', 'N/A')} | Page: {d.metadata.get('page', 'N/A')}\n{d.page_content}"
            for d in docs
        ]
    )

    prompt = f"""
You are an assistant answering questions using official QUEST documents.

Use ONLY the context provided below.
Answer the question clearly and concisely.
If the information is not explicitly present in the context, reply exactly:
"Not found in documents."

Context is related to:
QUEST – Quaid-e-Awam University of Engineering, Science and Technology, Nawabshah.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


# ----------------------------
# Main Chat
# ----------------------------
query = st.text_input("Ask a question from the documents")

if st.button("Ask"):
    if not api_key:
        st.error("Please provide API key")
    else:
        llm = load_llm(provider, model_name, api_key)
        answer = rag_answer(query, llm)
        st.subheader("Answer")
        st.write(answer)
