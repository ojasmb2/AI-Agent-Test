
import streamlit as st
import requests

st.set_page_config(page_title="Ask the LangGraph Agent (MCP)")

st.title("🎬 Ask the LangGraph Agent (MCP)")
st.write("Ask a question about the Sakila database:")

query = st.text_input(" ", placeholder="e.g. Which actor appears in the most films?")
sql_only = st.checkbox("Return SQL only", value=False)

if st.button("Send") and query:
    try:
        response = requests.post(
            "http://127.0.0.1:5001/mcp/invoke",  # must match your FastAPI server
            json={"query": query, "sql_only": sql_only},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            st.error(f"❌ Error: {data['error']}")
        else:
            st.success("✅ Response Received")
            st.code(data["sql"], language="sql")
            if not sql_only and "result" in data:
                st.write("📊 Result:")
                st.dataframe(data["result"])
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {str(e)}")