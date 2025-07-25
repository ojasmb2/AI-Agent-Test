from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage
from agent import build_graph  # same as in your test script
from dotenv import load_dotenv

load_dotenv()

# Build LangGraph agent
graph = build_graph(provider="ollama")

app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for security in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_agent(request: Request):
    body = await request.json()
    user_input = body.get("question", "")

    # Run LangGraph agent
    state_input = {"messages": [HumanMessage(content=user_input)]}
    response = graph.invoke(state_input)

    # Extract AI message from result
    ai_messages = [
        msg.content for msg in response.get("messages", [])
        if isinstance(msg, AIMessage)
    ]
    return {"response": ai_messages[-1] if ai_messages else "No AI response found."}