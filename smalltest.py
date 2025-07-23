import os
from agent import build_graph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# Build the graph for the Ollama provider
graph = build_graph(provider="ollama")

# Test input
query = "Which actor appears in the most films? Choose one of the films from that actor on random and list all of the other actors from that film."
state_input = {
    "messages": [HumanMessage(content=query)]
}

# Run the agent
response = graph.invoke(state_input)

print("\n==== AGENT RESPONSE ====")
print(response)