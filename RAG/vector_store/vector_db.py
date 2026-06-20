"""
load pdf
split document into chunks
create enbeddings 
store in chroma db
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

data= PyPDFLoader("/Users/amansinghpatel/Documents/program/langchain/RAG/vector store/Men-are-from-mars-women-are-from-venus.pdf")

docs = data.load()

# chunkking 

splitter= RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks= splitter.split_documents(docs)

# embedding
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector store
vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)