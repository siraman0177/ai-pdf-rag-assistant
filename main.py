from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from typer import prompt

load_dotenv()

embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma(
    persist_directory= "chroma_db",
    embedding_function= embedding_model
)

retriever= vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult": 0.5
    }
    
)
llm= ChatMistralAI(model= "mistral-small-latest")

# prompt Template
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", """
         you are a helpful ai assistant.
         use ONLY the provided context to answer the question.
         if the answer is not present in the context, 
         say: "i could not find the answer in the context provided," 
         """),
        ("human","""
         context:{context}
         
         question:{question}
         """)
    ]
)

print("RAG system is created")

print("0 to exit")

while True:
    query= input("you: ")
    if query=="0":
        print("exiting...")
        break
    docs = retriever.invoke(query)
    context= "\n".join([d.page_content for d in docs])
    
    final_prompt= prompt.invoke({
        "context":context,
        "question": query
    })
    
    response= llm.invoke(final_prompt)
    
    print(f"\n AI:{response.content}")
