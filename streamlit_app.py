import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent import build_graph  # your agent builder

# Build your graph
graph = build_graph(provider="ollama")

st.set_page_config(page_title="LangGraph Chat Agent")

st.title("Sakila Database Agent")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input
user_input = st.chat_input("Ask something about the Sakila DB...")

if user_input:
    # Add user message to memory
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Prepare input state for LangGraph
    state_input = {"messages": st.session_state.messages}

    # Run agent
    response = graph.invoke(state_input)

    # Get new assistant message
    ai_msg = [
        m for m in response.get("messages", [])
        if isinstance(m, AIMessage)
    ]
    if ai_msg:
        st.session_state.messages.append(ai_msg[-1])

# Display full chat
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)