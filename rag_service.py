from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load and Split Documents
def ingest_document(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks so the LLM doesn't get overwhelmed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # 2. Store in Vector Database
    vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings())
    return vectorstore

# 3. Create RAG Chain
def ask_document(vectorstore, query: str):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain.invoke(query)
