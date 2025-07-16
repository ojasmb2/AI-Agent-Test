import os
from agent import build_graph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# Build the graph for the Ollama provider
graph = build_graph(provider="ollama")

# Test input
query = "What was the largest drop in price of AAPL from closing to opening and what were the dates?"
state_input = {
    "messages": [HumanMessage(content=query)]
}

# Run the agent
response = graph.invoke(state_input)

print("\n==== AGENT RESPONSE ====")
print(response)