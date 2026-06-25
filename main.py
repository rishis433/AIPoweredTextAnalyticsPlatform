import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# 1. Setup App and LLM
app = FastAPI()
llm = ChatOpenAI(model="gpt-4o", temperature=0) # temperature 0 for consistent output

# 2. Define the desired output structure
response_schemas = [
    ResponseSchema(name="sentiment", description="The overall sentiment: Positive, Negative, or Neutral."),
    ResponseSchema(name="key_topics", description="A list of 3-5 main topics discussed in the text."),
    ResponseSchema(name="summary", description="A one-sentence summary of the text.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. Request Model
class AnalysisRequest(BaseModel):
    text: str

# 4. API Endpoint
@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    try:
        # Create the prompt with format instructions
        prompt = ChatPromptTemplate.from_template(
            "Analyze the following text and provide the results in the requested format.\n"
            "{format_instructions}\n"
            "Text: {text}"
        )
        
        # Build the chain
        chain = prompt | llm | output_parser
        
        # Execute
        result = chain.invoke({
            "format_instructions": output_parser.get_format_instructions(),
            "text": request.text
        })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
