import os
import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()
try:
    with open("system_prompt.txt", encoding="utf-8") as f:
        system_prompt = f.read().strip()
except FileNotFoundError:
    system_prompt = (
        "You are a helpful AI assistant that answers questions about the Sakila database. "
        "Use the provided tools: generate_sql to build SQL queries, and sql_executor to run them."
    )


def generate_sql(question: str) -> str:
    """Creates a SQL query for the Sakila database based on a user question."""
    if 'most films' in question.lower():
        return (
            "SELECT a.actor_id, a.first_name || ' ' || a.last_name AS actor, "
            "COUNT(fa.film_id) AS film_count "
            "FROM actor a "
            "JOIN film_actor fa ON a.actor_id = fa.actor_id "
            "GROUP BY a.actor_id "
            "ORDER BY film_count DESC "
            "LIMIT 1;"
        )
    # Fallback
    return f"SELECT * FROM film WHERE title LIKE '%{question}%';"


def sql_executor(sql: str) -> str:
    """Calls the MCP server to execute a SQL query and returns the formatted results."""
    try:
        resp = requests.post(
            "http://localhost:5001/mcp/execute_query/run", json={"query": sql}, timeout=10
        )
        resp.raise_for_status()
        result = resp.json().get("result", [])
        if not result:
            return "No rows returned."
        header = result[0].keys() if isinstance(result[0], dict) else []
        rows = [list(r.values()) for r in result] if isinstance(result[0], dict) else result
        lines = []
        if header:
            lines.append(" | ".join(header))
            lines.append("-|-" * len(header))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Execution error: {e}"

generate_sql_tool = Tool(
    name="generate_sql",
    func=generate_sql,
    description="Generate a SQL query for a Sakila database question"
)
sql_executor_tool = Tool(
    name="sql_executor",
    func=sql_executor,
    description="Execute a SQL query against the Sakila database via MCP server"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = initialize_agent(
    tools=[generate_sql_tool, sql_executor_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


def invoke(payload: dict) -> dict:
    """Receives a dict with {'messages': [<HumanMessage>...]} and returns {'messages': [...]}."""
    messages = payload.get("messages", [])
    user_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break
    if user_msg is None:
        return {"messages": [{"type": "ai", "content": "⚠️ No human message found."}]}

    try:
        output = agent.run(user_msg)
        return {"messages": [{"type": "ai", "content": output}]}
    except Exception as err:
        return {"messages": [{"type": "ai", "content": f"⚠️ Agent error: {err}"}]}