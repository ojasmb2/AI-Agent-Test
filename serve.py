from fastapi import FastAPI, Request
from langchain.schema import HumanMessage
from agent_graph import invoke  # your function-calling agent

# Alias for Uvicorn
fastapi_app = FastAPI()
app = fastapi_app  # launch with `uvicorn serve:app`

@fastapi_app.post("/")
async def route(request: Request):
    body = await request.json()
    user_input = body.get("input", "")

    # Invoke our agent and get back {'messages': [...]}
    result = invoke({"messages": [HumanMessage(content=user_input)]})

    # Pull out the messages list (fallback to empty list)
    messages = result.get("messages") or []

    # Return as top-level "messages"
    return {"messages": messages}