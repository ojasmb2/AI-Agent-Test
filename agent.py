import os
import datetime
import requests
import pytz
import yaml
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from tools.final_answer import FinalAnswerTool
from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool, load_tool, tool

# === TOOLS ===

@tool
def web_search(query: str) -> str:
    """Allows search through DuckDuckGo.
    Args:
        query: what you want to search
    """
    search_tool = DuckDuckGoSearchTool()
    results = search_tool(query)
    return "\n".join(results)

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def visit_webpage(url: str) -> str:
    """Fetches raw HTML content of a web page.
    Args:
        url: The url of the webpage.
    """
    try:
        response = requests.get(url, timeout=5)
        return response.text[:5000]  # Limit length
    except Exception as e:
        return f"[ERROR fetching {url}]: {str(e)}"

@tool
def text_splitter(text: str) -> List[str]:
    """Splits text into chunks using LangChain's CharacterTextSplitter.
    Args:
        text: A string of text to split.
    """
    splitter = CharacterTextSplitter(chunk_size=450, chunk_overlap=10)
    return splitter.split_text(text)

# === FINAL ANSWER TOOL ===
final_answer = FinalAnswerTool()

# === LOAD PROMPT TEMPLATES ===
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

# === LOAD agent.json CONFIG ===
with open("agent.json", "r") as f:
    agent_config = yaml.safe_load(f)

model_config = agent_config["model"]["data"]

# === BUILD MODEL ===
model = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.5,
    max_tokens=1024,
)

# === IMPORT TOOL FROM HUB ===
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# === BUILD AGENT ===
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        web_search,
        get_current_time_in_timezone,
        visit_webpage,
        text_splitter,
        image_generation_tool
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# === EXPORT AGENT ===
def get_agent():
    return agent
