import streamlit as st
import requests

st.set_page_config(page_title="Sakila SQL Agent", page_icon="🎬")
st.title("Sakila SQL Agent")

st.markdown("""
This assistant can:
- Generate SQL queries for the Sakila database
- Execute them **only if you say so**
""")

user_input = st.text_input("Ask a question about the Sakila database:", 
                          placeholder="e.g., Show all movies released in 2006")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        try:
            res = requests.post("http://localhost:8000", json={"input": user_input})
            res.raise_for_status()
            data = res.json()

            # Now read the flat "messages" array
            messages = data.get("messages", [])

            for msg in messages:
                # Our Agent returns plain dicts here
                role = msg.get("type", "")
                content = msg.get("content", "")

                if role == "human":
                    st.info(f"**You:** {content}")
                else:
                    st.success(f"**Assistant:**\n\n{content}")

        except Exception as e:
            st.error(f"Error: {e}")