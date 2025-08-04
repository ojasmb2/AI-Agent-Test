from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import build_graph

graph = build_graph(provider="ollama")

app = FastAPI()

class QueryInput(BaseModel):
    input: str

@app.post("/mcp/invoke")
async def run_agent(query: QueryInput):
    result = await graph.ainvoke({"messages": [{"role": "user", "content": query.input}]})
    return {"output": result}