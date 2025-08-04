#agent goes here
import os
import requests
import sqlite3
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
from langchain_core.messages import AIMessage

load_dotenv()

#Defualt operators:

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}


@tool
def sakila_sql_generator(query: str) -> str:
    """
    Generates a SQL query for the Sakila SQLite database based on a natural language question.
    Does NOT execute the query or access a database — it only returns the SQL code.
    
    Args:
        query: A natural language question about the Sakila DVD rental database.
        
    Returns:
        A string containing the SQL query.
    """
    system_prompt = """
You are an AI SQL assistant. You only generate valid SQL queries for the Sakila SQLite database based on user questions. 
Do not explain anything or return results. Just return the SQL query enclosed in triple backticks.

Sakila Schema (simplified):
- film(film_id, title, description, release_year, language_id, rental_rate, length, rating)
- actor(actor_id, first_name, last_name)
- film_actor(film_id, actor_id)
- customer(customer_id, first_name, last_name, address_id, store_id, email, active)
- rental(rental_id, rental_date, inventory_id, customer_id, return_date, staff_id)
- inventory(inventory_id, film_id, store_id)
- store(store_id, manager_staff_id, address_id)
- staff(staff_id, first_name, last_name, address_id, store_id)
- payment(payment_id, customer_id, staff_id, rental_id, amount, payment_date)
- category(category_id, name)
- film_category(film_id, category_id)

If the user question cannot be answered with this schema, return:
```sql
-- Cannot answer the question with the available Sakila schema
"""

@tool
def sakila_sql_executor(query: str) -> str:
    """
    Executes a SQL query on the local Sakila SQLite database and returns the result.
    
    Args:
        query: A SQL query string. Must be a valid SELECT statement.
    
    Returns:
        The result of the query as a JSON-style string or formatted table.
    """
    try:
        # Path to your local Sakila SQLite file
        db_path = "databases/sakila.db" 

        # Connect to SQLite and run the query
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        # Format results
        results = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        conn.close()

        # Return as JSON-like string
        if not results:
            return "Query executed successfully, but no results were returned."
        return str(results)

    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# load the system prompt from the file

import os

# Safely load system_prompt.txt
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are a helpful AI assistant."  # Default fallback prompt


# System message
sys_msg = SystemMessage(content=system_prompt)

# build a retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #  dim=768
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding= embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)



tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
    sakila_sql_generator,
    sakila_sql_executor
]

# Build graph function
def build_graph(provider: str = "ollama"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "google":
        # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0) # optional : qwen-qwq-32b gemma2-9b-it
    elif provider == "huggingface":
        # TODO: Add huggingface endpoint
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                #url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
                temperature=0,
            ),
        )
    elif provider == "ollama":
        llm = OllamaLLM(
            model="llama3",
            temperature=0,
        )

    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    # Bind tools to LLM
    # No longer needed with Ollama update
    #llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        response = llm.invoke(state["messages"])
    
        # Only return the SQL string — leave execution to the MCP server
        return {
            "messages": state["messages"] + [AIMessage(content=response.strip())]
        }

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
    
        if similar_question:
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}"
            )
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        else:
            # Fallback if no similar question is found
            return {"messages": [sys_msg] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

# test