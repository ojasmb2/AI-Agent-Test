# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# --- PDF uploader and parser ---
def parse_pdfs(uploaded_files):
    pdf_docs = []
    for uploaded_file in uploaded_files:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            pdf_docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
    return pdf_docs

# --- Guest info retrieval ---
def build_retriever(all_docs):
    return BM25Retriever.from_documents(all_docs)

def extract_text(query: str, retriever):
    results = retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "لم يتم العثور على معلومات مطابقة في الملفات."

# --- Streamlit UI ---
st.set_page_config(page_title="NINU Agent", page_icon="🏛️")
st.title("🏛️ NINU - Guest & PDF & Web Assistant")

st.markdown("** Hint:** NINU can help summarize lectures, answer questions from PDFs, and search the web interactively.")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

query = st.text_area("📝 اكتب سؤالك أو كمل مذاكرتك هنا:")

uploaded_files = st.file_uploader("📄 ارفع ملفات PDF للمحاضرات", type=["pdf"], accept_multiple_files=True)

if st.button("Ask NINU") and query:
    # Parse PDFs if uploaded
    user_docs = parse_pdfs(uploaded_files) if uploaded_files else []
    bm25_retriever = build_retriever(user_docs) if user_docs else None

    # Tool for PDF retrieval (if PDFs uploaded)
    def pdf_tool_func(q):
        if bm25_retriever:
            return extract_text(q, bm25_retriever)
        else:
            return "لا توجد ملفات PDF مرفوعة للبحث."

    NINU_tool = Tool(
        name="NINU_Lec_retriever",
        func=pdf_tool_func,
        description="Retrieves content from uploaded PDFs based on a query."
    )

    # Tool for Web search using SerpAPI
    serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    SerpAPI_tool = Tool(
        name="WebSearch",
        func=serpapi.run,
        description="Searches the web for recent information."
    )

    # Combine tools
    tools = [NINU_tool, SerpAPI_tool]

    # Create LLM and bind tools
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=groq_api_key)
    llm_with_tools = llm.bind_tools(tools)

    # Define Agent state and assistant function
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    def assistant(state: AgentState):
        return {
            "messages": [llm_with_tools.invoke(state["messages"])]
        }

    # Build the StateGraph agent
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    NINU = builder.compile()

    # Add intro prompt if first message
    if len(st.session_state.conversation_history) == 0:
        intro_prompt = """
You are a general AI assistant with access to two tools:

1. NINU_Lec_retriever: retrieves content from uploaded PDFs based on a query.
2. WebSearch: performs web searches to answer questions about current events or general knowledge.

Based on the user's query, decide whether to use NINU_Lec_retriever, WebSearch, or both.

When answering, report your thoughts and finish your answer with the following template:
FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use commas or units (like $, %, etc.) unless specified.

If you are asked for a string, avoid articles, abbreviations, and write digits in plain text unless specified.
"""
        st.session_state.conversation_history.append(HumanMessage(content=intro_prompt))

    # Add user query
    st.session_state.conversation_history.append(HumanMessage(content=query))

    # Invoke the agent
    response = NINU.invoke({"messages": st.session_state.conversation_history})

    # Append assistant reply to conversation history
    assistant_reply = response["messages"][-1]
    st.session_state.conversation_history.append(assistant_reply)

    # Show assistant reply
    st.markdown("###  NINU's Response:")
    st.write(assistant_reply.content)

    # Show full conversation history (optional)
    with st.expander("🧾 Show full conversation history"):
        for msg in st.session_state.conversation_history:
            role = "You" if msg.type == "human" else "NINU"
            st.markdown(f"**{role}:** {msg.content}")
