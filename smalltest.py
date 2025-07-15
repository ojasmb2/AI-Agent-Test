import os
from agent import build_graph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# Build the graph for the Ollama provider
graph = build_graph(provider="ollama")

# Test input
query = "What is the article by Carolyn Collins Petersen about?"
state_input = {
    "messages": [HumanMessage(content=query)]
}

# Run the agent
response = graph.invoke(state_input)

print("\n==== AGENT RESPONSE ====")
print(response)