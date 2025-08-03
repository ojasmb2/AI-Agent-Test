import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent import build_graph  # from your agent.py

st.set_page_config(page_title="LangGraph Agent", layout="centered")

# Build your LangGraph graph once
graph = build_graph(provider="ollama")

st.title("🎬 LangGraph Agent - Sakila DB")

user_input = st.text_input("Ask a question about the Sakila database:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Submit") and user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # LangGraph expects state input like this:
    state_input = {"messages": st.session_state.chat_history}

    # Run the agent
    response = graph.invoke(state_input)

    # Extract the new AI message
    ai_messages = [
        msg for msg in response.get("messages", [])
        if isinstance(msg, AIMessage)
    ]
    if ai_messages:
        st.session_state.chat_history.append(ai_messages[-1])

# Display the conversation
for msg in st.session_state.chat_history:
    role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Agent"
    st.markdown(f"**{role}:** {msg.content}")