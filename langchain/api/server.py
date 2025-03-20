import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Simple Langchain Server"
)

# Get API key and model type from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
LLM_TYPE_MODEL = os.getenv("LLM_MODEL")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables")

if not LLM_TYPE_MODEL:
    raise ValueError("LLM_MODEL needs to have a valid model type")


# Define Pydantic response model (if needed for responses)
class ChatBaseResponse(BaseModel):
    response: str


# Initialize Gemini API
llm = ChatGoogleGenerativeAI(
    model=LLM_TYPE_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define Prompt Templates
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Your name is Sarah."),
    ("human", "{user_input}")
])

# Create a Runnable Chain for LangServe
def generate_prompt(user_input: str):
    return prompt_template.format(user_input=user_input)

chain = RunnableLambda(generate_prompt) | llm

# Add LangServe routes
add_routes(app, chain, path="/gemini")


@app.get("/gemini", response_model=ChatBaseResponse)
async def gemini_route():
    user_prompt = "Hello friend, whats your name?"
    ai_msg = llm.invoke(user_prompt)  # Expecting a single prompt, not a list
    return {"response": ai_msg.content}


# Start Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
