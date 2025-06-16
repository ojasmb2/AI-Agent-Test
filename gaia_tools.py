from smolagents import PythonInterpreterTool, tool
import requests

@tool
def ReverseTextTool(text: str) -> str:
    """
    Reverses a text string character by character.
    Args:
        text (str): The text to reverse
    Returns:
        str: The reversed text
    """
    return text[::-1]


@tool
def RunPythonFileTool(file_path: str) -> str:
    """
    Executes a Python script loaded from the specified path using the PythonInterpreterTool.
    Args:
        file_path (str): The full path to the python (.py) file containing the Python code.
    Returns:
        str: The output produced by the code execution, or an error message if it fails.
    """
    try:
        with open(file_path, "r") as f:
            code = f.read()
        interpreter = PythonInterpreterTool()
        result = interpreter.run({"code": code})
        return result.get("output", "No output returned.")
    except Exception as e:
        return f"Execution failed: {e}"

@tool
def download_server(url: str, save_path: str) -> str:
    """
    Downloads a file from a URL and saves it to the given path.
    Args:
        url (str): The URL from which to download the file.
        save_path (str): The local file path where the downloaded file will be saved.
    Returns:
        str: A message indicating the result of the download operation.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return f"File downloaded to {save_path}"
    except Exception as e:
        return f"Failed to download: {e}"