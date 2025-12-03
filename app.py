import os
import streamlit as st

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Vijayawada Health Assistant",
    page_icon="🩺",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Center the chat and give WhatsApp-ish look */
    .main > div {
        max-width: 700px;
        margin: 0 auto;
    }
    .user-bubble {
        background-color: #dcf8c6;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        display: inline-block;
    }
    .bot-bubble {
        background-color: #ffffff;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 4px 0;
        display: inline-block;
        border: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# LOAD VECTORSTORE (CACHED)
# -----------------------------
@st.cache_resource
def get_vectorstore():
    """
    Build FAISS index from CSV on first run, then reuse.
    """
    csv_path = "vijayawada_health_full_merged.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Please ensure it's in the same folder as app.py."
        )

    # 1. Load CSV
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # 3. Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 4. Build FAISS index
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore, embedding_model


# -----------------------------
# LOAD LLM (CACHED)
# -----------------------------
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# -----------------------------
# RAG ANSWER FUNCTION
# -----------------------------
def get_answer(question: str, vectorstore: FAISS, llm: HuggingFacePipeline) -> str:
    # Retrieve relevant docs
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful health assistant for Vijayawada.
Use ONLY the context below and answer the question in a clear, short way.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        if isinstance(response, str):
            return response.strip()
        return str(response)
    except Exception as e:
        return f"Sorry, I faced an error while generating the answer: {e}"


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("🩺 Vijayawada Health Assistant")
    st.write("Ask about **hospitals, clinics, health services in Vijayawada** based on the provided data.")

    # Load models/resources once
    with st.spinner("Loading models and vector index (first time may take a bit)..."):
        vectorstore, _ = get_vectorstore()
        llm = get_llm()

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I’m your Vijayawada health assistant. How can I help you today?"}
        ]

    # Display chat history (WhatsApp-style bubbles)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # Chat input at bottom
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

        # Generate bot reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_answer(user_input, vectorstore, llm)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(f"<div class='bot-bubble'>{reply}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
