import requests
import pandas as pd
from PIL import Image
import os
import subprocess
from bs4 import BeautifulSoup
import urllib.parse

def web_search(query: str, api_key: str = None) -> str:
    """
    Perform web search using SerpAPI if available, otherwise fallback to DuckDuckGo scraping.
    """
    if api_key and api_key != "your-serpapi-key-here":
        return _serpapi_search(query, api_key)
    else:
        return _duckduckgo_search(query)

def _serpapi_search(query: str, api_key: str) -> str:
    """Search using SerpAPI."""
    try:
        url = f"https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        results = response.json()
        organic_results = results.get("organic_results", [])
        
        if organic_results:
            # Get top 3 results
            search_summary = []
            for i, result in enumerate(organic_results[:3]):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                if title and snippet:
                    search_summary.append(f"{i+1}. {title}: {snippet}")
            
            return "\n".join(search_summary) if search_summary else "No useful results found"
        else:
            return "No search results found"
            
    except requests.RequestException as e:
        print(f"SerpAPI search error: {e}")
        return "Search failed"

def _duckduckgo_search(query: str) -> str:
    """Fallback web search using DuckDuckGo scraping."""
    try:
        # DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Try to get instant answer
        abstract = data.get("Abstract", "")
        if abstract:
            return f"Summary: {abstract}"
        
        # Try related topics
        related_topics = data.get("RelatedTopics", [])
        if related_topics:
            summaries = []
            for topic in related_topics[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    summaries.append(topic["Text"])
            if summaries:
                return "Related information:\n" + "\n".join(summaries)
        
        # Fallback to web scraping (simplified)
        return _simple_web_scrape(query)
        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return "Search failed"

def _simple_web_scrape(query: str) -> str:
    """Simple web scraping fallback."""
    try:
        # Use a simple search approach
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Try to extract some basic information
            results = soup.find_all('a', class_='result__snippet')[:3]
            if results:
                snippets = [r.get_text().strip() for r in results if r.get_text().strip()]
                return "\n".join(snippets[:3]) if snippets else "Limited search results available"
        
        return "Basic web search completed - limited results"
        
    except Exception as e:
        print(f"Web scraping error: {e}")
        return "Web search unavailable"

def read_file(file_name: str) -> str:
    """
    Read and process different file types (text, CSV, images).
    """
    if not file_name or not os.path.exists(file_name):
        return "File not found"
    
    try:
        file_extension = os.path.splitext(file_name)[1].lower()
        
        if file_extension == ".csv":
            return _read_csv_file(file_name)
        elif file_extension in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
            return _read_image_file(file_name)
        elif file_extension in [".txt", ".md", ".py", ".js", ".html", ".json"]:
            return _read_text_file(file_name)
        else:
            # Try to read as text file
            return _read_text_file(file_name)
            
    except Exception as e:
        return f"Error reading file: {str(e)}"

def _read_text_file(file_name: str) -> str:
    """Read a text file."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:5000]  # Limit to first 5000 characters
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_name, "r", encoding="latin-1") as f:
                content = f.read()
            return content[:5000]
        except Exception as e:
            return f"Text file reading error: {str(e)}"

def _read_csv_file(file_name: str) -> str:
    """Read and summarize a CSV file."""
    try:
        df = pd.read_csv(file_name)
        
        # Create a summary
        summary = []
        summary.append(f"CSV file shape: {df.shape[0]} rows, {df.shape[1]} columns")
        summary.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Show first few rows
        summary.append("\nFirst 5 rows:")
        summary.append(df.head().to_string())
        
        # Show basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            summary.append(f"\nNumeric column statistics:")
            summary.append(df[numeric_columns].describe().to_string())
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"CSV reading error: {str(e)}"

def _read_image_file(file_name: str) -> str:
    """Read and analyze an image file."""
    try:
        # Try OCR first
        try:
            import pytesseract
            img = Image.open(file_name)
            
            # Get image info
            info = f"Image: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}"
            
            # Try OCR
            text = pytesseract.image_to_string(img).strip()
            if text:
                return f"{info}\n\nExtracted text:\n{text}"
            else:
                return f"{info}\n\nNo text detected in image."
                
        except ImportError:
            # OCR not available, just return image info
            img = Image.open(file_name)
            return f"Image: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}\n(OCR not available - install pytesseract for text extraction)"
            
    except Exception as e:
        return f"Image reading error: {str(e)}"

def execute_code(code: str, timeout: int = 10) -> str:
    """
    Execute Python code safely with timeout.
    """
    try:
        # Basic security check - prevent dangerous operations
        dangerous_keywords = ["import os", "import subprocess", "__import__", "exec", "eval", "open("]
        if any(keyword in code.lower() for keyword in dangerous_keywords):
            return "Code execution blocked: potentially unsafe operations detected"
        
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp"  # Run in safe directory
        )
        
        if result.returncode == 0:
            return result.stdout.strip() if result.stdout else "Code executed successfully (no output)"
        else:
            return f"Code execution error: {result.stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        return "Code execution timeout"
    except Exception as e:
        return f"Code execution error: {str(e)}"

def calculate_simple_math(expression: str) -> str:
    """
    Safely evaluate simple mathematical expressions.
    """
    try:
        # Only allow basic math characters
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Invalid mathematical expression"
        
        # Use eval safely for basic math
        result = eval(expression)
        return str(result)
        
    except Exception as e:
        return f"Math calculation error: {str(e)}" 