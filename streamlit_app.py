import streamlit as st
import requests

# Backend API endpoint
API_URL = "http://localhost:8000/query"  # Change this to your actual endpoint

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")

st.title("🧠 LangGraph Agent Interface")
st.markdown("Ask your question and the AI agent will respond using the Sakila database!")

# Input field
user_question = st.text_input("Your question:")

# Button
if st.button("Submit") and user_question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": user_question})
            if response.status_code == 200:
                result = response.json()
                # You might need to adjust this depending on your API format
                st.success("Agent response:")
                st.code(result.get("response", "No 'response' field in API result."))
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")