# smol_agent.py

import os
from dotenv import load_dotenv
from smolagents import CodeAgent, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ============== Arithmetic Tools ==============

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get modulus of two numbers."""
    return a % b

# ============== Wikipedia Search Tool ==============

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return max 2 results."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

# ============== Tavily Search Tool ==============

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return max 3 results."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

# ============== Arxiv Search Tool ==============

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return max 3 results."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return formatted_search_docs

# ============== Retriever Tool ==============

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_SERVICE_KEY")
)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)

@tool
def retrieve_similar_question(query: str) -> str:
    """Retrieve similar question from vector store."""
    similar_doc = vector_store.similarity_search(query, k=1)[0]
    content = similar_doc.page_content
    if "Final answer :" in content:
        answer = content.split("Final answer :")[-1].strip()
    else:
        answer = content.strip()
    return answer

# ============== Load System Prompt ==============

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# ============== Initialize LLM ==============

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# ============== Build SmolAgent ==============

agent = CodeAgent(
    llm=llm,
    tools=[
        multiply,
        add,
        subtract,
        divide,
        modulus,
        wiki_search,
        web_search,
        arxiv_search,
        retrieve_similar_question
    ],
    system_prompt=system_prompt
)

# ============== Example Run ==============

if __name__ == "__main__":
    user_query = "Find recent arxiv papers on diffusion models."
    response = agent.run(user_query)
    print(response)
