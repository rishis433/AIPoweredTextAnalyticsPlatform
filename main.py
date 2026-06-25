import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# --- 1. Database Setup ---
DATABASE_URL = "sqlite:///./analytics.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisRecord(Base):
    __tablename__ = "analytics"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    analysis_result = Column(JSON)

Base.metadata.create_all(bind=engine)

# --- 2. RAG & LLM Setup ---
# Assume you have a folder named 'docs' with your PDFs
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
llm = ChatOpenAI(model="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# --- 3. FastAPI App ---
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

@app.post("/ask")
async def ask_and_save(request: QueryRequest):
    try:
        # A. Execute RAG
        response = qa_chain.invoke({"query": request.user_query})
        result_text = response["result"]

        # B. Persist in Database
        db = SessionLocal()
        new_entry = AnalysisRecord(query=request.user_query, analysis_result={"answer": result_text})
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        db.close()

        return {"id": new_entry.id, "answer": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
