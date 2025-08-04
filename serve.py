from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3

from agent import build_graph  # make sure this points to your LangGraph agent

app = FastAPI()

# Build the LangGraph agent once on startup
agent_executor = build_graph(provider="ollama")

# Request schema
class AgentRequest(BaseModel):
    query: str
    sql_only: bool = False  # If true, return only the SQL and don't execute

# Core handler for processing user queries
@app.post("/agent")
async def run_agent(request: AgentRequest):
    user_msg = {"messages": [{"role": "user", "content": request.query}]}
    result = agent_executor.invoke(user_msg)

    import re

    raw_response = result["messages"][-1].content

    # Extract the first SQL block from the assistant's output
    sql_match = re.search(r"(?i)(select|with)\s.+?;\s", raw_response, re.DOTALL)
    sql_cleaned = sql_match.group(0).strip() if sql_match else ""

    print("🔍 Cleaned SQL being executed:\n", sql_cleaned)

    # If user wants only the SQL
    if request.sql_only:
        return {"sql": sql_cleaned}

    # Otherwise execute the cleaned SQL
    try:
        conn = sqlite3.connect("databases/sakila.db")
        cursor = conn.cursor()
        cursor.execute(sql_cleaned)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result_data = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return {"sql": sql_cleaned, "result": result_data}
    except Exception as e:
        return {"sql": sql_cleaned, "error": str(e)}

# Alias route to support frontend that calls /mcp/invoke
@app.post("/mcp/invoke")
async def mcp_invoke(request: AgentRequest):
    return await run_agent(request)

# Optional if running directly with `python serve.py`
if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=5001, reload=True)