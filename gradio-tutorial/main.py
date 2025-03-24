import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
import gradio as gr

load_dotenv()

#llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022",streaming=True)

# Get API key and model type from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
LLM_TYPE_MODEL = os.getenv("LLM_MODEL")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables")

if not LLM_TYPE_MODEL:
    raise ValueError("LLM_MODEL needs to have a valid model type")



# Initialize Gemini API
llm = ChatGoogleGenerativeAI(
    model=LLM_TYPE_MODEL,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    streaming=True
)


system_message = "you act like an astronaut"

def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content=system_message))

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    if message is not None:
        history_langchain_format.append(HumanMessage(content=message))
        partial_message = ""
        for response in llm.stream(history_langchain_format):
            partial_message += response.content
            yield partial_message


demo_interface = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
                       container=False,
                       autoscroll=True,
                       scale=7),
)

demo_interface.launch(debug=True, share=True)