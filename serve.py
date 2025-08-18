from fastapi import FastAPI, Request
from langchain.schema import HumanMessage
from agent_graph import invoke

fastapi_app = FastAPI()
app = fastapi_app

@fastapi_app.post("/")
async def route(request: Request):
    body = await request.json()
    user_input = body.get("input", "")

    result = invoke({"messages": [HumanMessage(content=user_input)]})

    messages = result.get("messages") or []

    return {"messages": messages}