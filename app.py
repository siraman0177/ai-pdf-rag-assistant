import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------
# Custom CSS
# --------------------

st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

.chat-box {
    padding:15px;
    border-radius:10px;
    background-color:#262730;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="big-title">📚 AI PDF Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Chat with your PDFs using Mistral AI</div>',
    unsafe_allow_html=True
)

# --------------------
# Models
# --------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatMistralAI(
    model="mistral-small-latest"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:

"I could not find the answer in the provided context."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

# --------------------
# Sidebar
# --------------------

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf"
    )

    process_btn = st.button("Process Document")

# --------------------
# PDF Processing
# --------------------

if process_btn and uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Loading PDF..."):

        loader = PyPDFLoader(file_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state["db_ready"] = True

    st.success("PDF processed successfully!")

# --------------------
# Chat
# --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("db_ready"):

    question = st.chat_input(
        "Ask something about the document..."
    )

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(question)

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(final_prompt)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])