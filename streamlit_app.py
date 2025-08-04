
import streamlit as st
import requests
import json

# Backend MCP server endpoint
MCP_ENDPOINT = "http://127.0.0.1:5001/mcp/invoke"

st.set_page_config(page_title="Sakila Agent", layout="centered")
st.title("🎬 Ask the LangGraph Agent (MCP)")

# Input field
query = st.text_input("Ask a question about the Sakila database:")

# Send to agent on submit
if st.button("Send") and query:
    try:
        with st.spinner("Thinking..."):
            res = requests.post(
                MCP_ENDPOINT,
                json={"input": query},
                timeout=30
            )
            res.raise_for_status()
            output = res.json().get("output", {})

            st.markdown("### 🤖 Agent Response:")

            if isinstance(output, list):
                for msg in output:
                    role = msg.get("role", "assistant").capitalize()
                    content = msg.get("content", "")
                    emoji = "🧑" if role.lower() == "user" else "🤖"
                    st.markdown(f"{emoji} **{role}:** {content}")
            elif isinstance(output, (dict, list)):
                st.json(output)
            else:
                st.markdown(f"> {output}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
