import os
import tempfile
import time
import re
import json
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import requests
import yt_dlp
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.agents import Tool, AgentExecutor, ConversationalAgent, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools import BaseTool, Tool, tool
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import google.generativeai as genai
from pydantic import Field

from smolagents import WikipediaSearchTool

class SmolagentToolWrapper(BaseTool):
    """Wrapper for smolagents tools to make them compatible with LangChain."""
    
    wrapped_tool: object = Field(description="The wrapped smolagents tool")
    
    def __init__(self, tool):
        """Initialize the wrapper with a smolagents tool."""
        super().__init__(
            name=tool.name,
            description=tool.description,
            return_direct=False,
            wrapped_tool=tool
        )

    def _run(self, query: str) -> str:
        """Use the wrapped tool to execute the query."""
        try:
            # For WikipediaSearchTool
            if hasattr(self.wrapped_tool, 'search'):
                return self.wrapped_tool.search(query)
            # For DuckDuckGoSearchTool and others
            return self.wrapped_tool(query)
        except Exception as e:
            return f"Error using tool: {str(e)}"
    
    def _arun(self, query: str) -> str:
        """Async version - just calls sync version since smolagents tools don't support async."""
        return self._run(query)

class WebSearchTool:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum time between requests in seconds
        self.max_retries = 10

    def search(self, query: str, domain: Optional[str] = None) -> str:
        """Perform web search with rate limiting and retries."""
        for attempt in range(self.max_retries):
            # Implement rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            try:
                # Make the search request
                results = self._do_search(query, domain)
                self.last_request_time = time.time()
                return results
            except Exception as e:
                if "202 Ratelimit" in str(e):
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        wait_time = (2 ** attempt) * self.min_request_interval
                        time.sleep(wait_time)
                        continue
                return f"Search failed after {self.max_retries} attempts: {str(e)}"
        
        return "Search failed due to rate limiting"

    def _do_search(self, query: str, domain: Optional[str] = None) -> str:
        """Perform the actual search request."""
        try:
            # Construct search URL
            base_url = "https://html.duckduckgo.com/html"
            params = {"q": query}
            if domain:
                params["q"] += f" site:{domain}"

            # Make request with increased timeout
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            if response.status_code == 202:
                raise Exception("202 Ratelimit")

            # Extract search results
            results = []
            soup = BeautifulSoup(response.text, 'html.parser')
            for result in soup.find_all('div', {'class': 'result'}):
                title = result.find('a', {'class': 'result__a'})
                snippet = result.find('a', {'class': 'result__snippet'})
                if title and snippet:
                    results.append({
                        'title': title.get_text(),
                        'snippet': snippet.get_text(),
                        'url': title.get('href')
                    })

            # Format results
            formatted_results = []
            for r in results[:10]:  # Limit to top 5 results
                formatted_results.append(f"[{r['title']}]({r['url']})\n{r['snippet']}\n")

            return "## Search Results\n\n" + "\n".join(formatted_results)

        except requests.RequestException as e:
            raise Exception(f"Search request failed: {str(e)}")

def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a temporary file and return the path.
    Useful for processing files from the GAIA API.
    
    Args:
        content: The content to save to the file
        filename: Optional filename, will generate a random name if not provided
        
    Returns:
        Path to the saved file
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)
    
    # Write content to the file
    with open(filepath, 'w') as f:
        f.write(content)
    
    return f"File saved to {filepath}. You can read this file to process its contents."


def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    
    Args:
        url: The URL to download from
        filename: Optional filename, will generate one based on URL if not provided
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                # Generate a random name if we couldn't extract one
                import uuid
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"File downloaded to {filepath}. You can now process this file."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract (if available).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text or error message
    """
    try:
        # Try to import pytesseract
        import pytesseract
        from PIL import Image
        
        # Open the image
        image = Image.open(image_path)
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        return f"Extracted text from image:\n\n{text}"
    except ImportError:
        return "Error: pytesseract is not installed. Please install it with 'pip install pytesseract' and ensure Tesseract OCR is installed on your system."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the CSV file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the Excel file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Run various analyses based on the query
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*will be deprecated.*")
        warnings.filterwarnings("ignore", "LangChain.*")
        
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the LLM
        self.llm = self._setup_llm()
        
        # Setup tools
        self.tools = [
            SmolagentToolWrapper(WikipediaSearchTool()),
            Tool(
                name="analyze_video",
                func=self._analyze_video,
                description="Analyze YouTube video content directly"
            ),
            Tool(
                name="analyze_image",
                func=self._analyze_image,
                description="Analyze image content"
            ),
            Tool(
                name="analyze_table",
                func=self._analyze_table,
                description="Analyze table or matrix data"
            ),
            Tool(
                name="analyze_list",
                func=self._analyze_list,
                description="Analyze and categorize list items"
            ),
            Tool(
                name="web_search",
                func=self._web_search,
                description="Search the web for information"
            )
        ]
        
        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent
        self.agent = self._setup_agent()
       

    def run(self, query: str) -> str:
        """Run the agent on a query with incremental retries."""
        max_retries = 3
        base_sleep = 1  # Start with 1 second sleep
        
        for attempt in range(max_retries):
            try:

                # If no match found in answer bank, use the agent
                response = self.agent.run(query)
                return response

            except Exception as e:
                sleep_time = base_sleep * (attempt + 1)  # Incremental sleep: 1s, 2s, 3s
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                return f"Error processing query after {max_retries} attempts: {str(e)}"

        print("Agent processed all queries!")

    def _clean_response(self, response: str) -> str:
        """Clean up the response from the agent."""
        # Remove any tool invocation artifacts
        cleaned = re.sub(r'> Entering new AgentExecutor chain...|> Finished chain.', '', response)
        cleaned = re.sub(r'Thought:.*?Action:.*?Action Input:.*?Observation:.*?\n', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def run_interactive(self):
        print("AI Assistant Ready! (Type 'exit' to quit)")
        
        while True:
            query = input("You: ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            print("Assistant:", self.run(query))

    def _web_search(self, query: str, domain: Optional[str] = None) -> str:
        """Perform web search with rate limiting and retries."""
        try:
            # Use DuckDuckGo API wrapper for more reliable results
            search = DuckDuckGoSearchAPIWrapper(max_results=5)
            results = search.run(f"{query} {f'site:{domain}' if domain else ''}")
            
            if not results or results.strip() == "":
                return "No search results found."
                
            return results

        except Exception as e:
            return f"Search error: {str(e)}"

    def _analyze_video(self, url: str) -> str:
        """Analyze video content using Gemini's video understanding capabilities."""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return "Please provide a valid video URL with http:// or https:// prefix."
            
            # Check if it's a YouTube URL
            if 'youtube.com' not in url and 'youtu.be' not in url:
                return "Only YouTube videos are supported at this time."

            try:
                # Configure yt-dlp with minimal extraction
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'no_playlist': True,
                    'youtube_include_dash_manifest': False
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        # Try basic info extraction
                        info = ydl.extract_info(url, download=False, process=False)
                        if not info:
                            return "Could not extract video information."

                        title = info.get('title', 'Unknown')
                        description = info.get('description', '')
                        
                        # Create a detailed prompt with available metadata
                        prompt = f"""Please analyze this YouTube video:
Title: {title}
URL: {url}
Description: {description}

Please provide a detailed analysis focusing on:
1. Main topic and key points from the title and description
2. Expected visual elements and scenes
3. Overall message or purpose
4. Target audience"""

                        # Use the LLM with proper message format
                        messages = [HumanMessage(content=prompt)]
                        response = self.llm.invoke(messages)
                        return response.content if hasattr(response, 'content') else str(response)

                    except Exception as e:
                        if 'Sign in to confirm' in str(e):
                            return "This video requires age verification or sign-in. Please provide a different video URL."
                        return f"Error accessing video: {str(e)}"

            except Exception as e:
                return f"Error extracting video info: {str(e)}"

        except Exception as e:
            return f"Error analyzing video: {str(e)}"

    def _analyze_table(self, table_data: str) -> str:
        """Analyze table or matrix data."""
        try:
            if not table_data or not isinstance(table_data, str):
                return "Please provide valid table data for analysis."

            prompt = f"""Please analyze this table:

{table_data}

Provide a detailed analysis including:
1. Structure and format
2. Key patterns or relationships
3. Notable findings
4. Any mathematical properties (if applicable)"""

            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            return f"Error analyzing table: {str(e)}"

    def _analyze_image(self, image_data: str) -> str:
        """Analyze image content."""
        try:
            if not image_data or not isinstance(image_data, str):
                return "Please provide a valid image for analysis."

            prompt = f"""Please analyze this image:

{image_data}

Focus on:
1. Visual elements and objects
2. Colors and composition
3. Text or numbers (if present)
4. Overall context and meaning"""

            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def _analyze_list(self, list_data: str) -> str:
        """Analyze and categorize list items."""
        if not list_data:
            return "No list data provided."
        try:
            items = [x.strip() for x in list_data.split(',')]
            if not items:
                return "Please provide a comma-separated list of items."
            # Add list analysis logic here
            return "Please provide the list items for analysis."
        except Exception as e:
            return f"Error analyzing list: {str(e)}"

    def _setup_llm(self):
        """Set up the language model."""
        # Set up model with video capabilities
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 2000,
            "candidate_count": 1,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0,
            max_output_tokens=2000,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_message=SystemMessage(content=(
                "You are a precise AI assistant that helps users find information and analyze content. "
                "You can directly understand and analyze YouTube videos, images, and other content. "
                "When analyzing videos, focus on relevant details like dialogue, text, and key visual elements. "
                "For lists, tables, and structured data, ensure proper formatting and organization. "
                "If you need additional context, clearly explain what is needed."
            ))
        )
        
    def _setup_agent(self) -> AgentExecutor:
        """Set up the agent with tools and system message."""
        
        # Define the system message template
        PREFIX = """You are a helpful AI assistant that can use various tools to answer questions and analyze content. You have access to tools for web search, Wikipedia lookup, and multimedia analysis.

TOOLS:
------
You have access to the following tools:"""

        FORMAT_INSTRUCTIONS = """To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin! Remember to ALWAYS include 'Thought:', 'Action:', 'Action Input:', and 'Final Answer:' in your responses."""

        SUFFIX = """Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""

        # Create the base agent
        agent = ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            prefix=PREFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            suffix=SUFFIX,
            input_variables=["input", "chat_history", "agent_scratchpad", "tool_names"],
            handle_parsing_errors=True
        )

        # Initialize agent executor with custom output handling
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            max_iterations=5,
            verbose=True,
            handle_parsing_errors=True,
            return_only_outputs=True  # This ensures we only get the final output
        )

@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the CSV file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas is not installed. Please install it with 'pip install pandas'."
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    
    Args:
        file_path: Path to the Excel file
        query: Question about the data
        
    Returns:
        Analysis result or error message
    """
    try:
        import pandas as pd
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Run various analyses based on the query
        result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"
        
        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())
        
        return result
    except ImportError:
        return "Error: pandas and openpyxl are not installed. Please install them with 'pip install pandas openpyxl'."
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"
