import os
import gradio as gr
import requests
import pandas as pd

from smolagents import LiteLLMModel, CodeAgent, DuckDuckGoSearchTool
from gaia_tools import ReverseTextTool, RunPythonFileTool, download_server

# System prompt for the agent
SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question.
Report your thoughts, and finish your answer with just the answer — no prefixes like "FINAL ANSWER:".
Your answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you're asked for a number, don’t use commas or units like $ or %, unless specified.
If you're asked for a string, don’t use articles or abbreviations (e.g. for cities), and write digits in plain text unless told otherwise.

Tool Use Guidelines:
1. Do *not* use any tools outside of the provided tools list.
2. Always use *only one tool at a time* in each step of your execution.
3. If the question refers to a .py file or uploaded Python script, use *RunPythonFileTool* to execute it and base your answer on its output.
4. If the question looks reversed (starts with a period or reads backward), first use *ReverseTextTool* to reverse it, then process the question.
5. For logic or word puzzles, solve them directly unless they are reversed — in which case, decode first using *ReverseTextTool*.
6. When dealing with Excel files, prioritize using the *excel* tool over writing code in *terminal-controller*.
7. If you need to download a file, always use the *download_server* tool and save it to the correct path.
8. Even for complex tasks, assume a solution exists. If one method fails, try another approach using different tools.
9. Due to context length limits, keep browser-based tasks (e.g., searches) as short and efficient as possible.
"""
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Agent wrapper using LiteLLMModel
class MyAgent:
    def _init_(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")
        
        self.model = LiteLLMModel(
            model_id="gemini/gemini-2.0-flash-lite",
            api_key=gemini_api_key,
            system_prompt=SYSTEM_PROMPT
        )
        
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                ReverseTextTool,
                RunPythonFileTool,
                download_server
            ],
            model=self.model,
            add_base_tools=True,
        )

    def _call_(self, question: str) -> str:
        return self.agent.run(question)

# Main evaluation function
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please login to Hugging Face.", None

    questions_url = f"{DEFAULT_API_URL}/questions"
    submit_url = f"{DEFAULT_API_URL}/submit"

    try:
        agent = MyAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None

results_log = []
answers_payload = []

for item in questions_data:
    task_id = item.get("task_id")
    question_text = item.get("question")
    if not task_id or question_text is None:
        continue
    try:
        submitted_answer = agent(question_text)
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
    except Exception as e:
        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Submitted Answer": f"AGENT ERROR: {e}"
        })

if not answers_payload:
    return "Agent did not return any answers.", pd.DataFrame(results_log)

    submission_data = {
        "username": profile.username.strip(),
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers_payload
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission failed: {e}", pd.DataFrame(results_log)

# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown("""
    *Instructions:*
    1. Clone this space and configure your Gemini API key.
    2. Log in to Hugging Face.
    3. Run your agent on evaluation tasks and submit answers.
    """)

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Results", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("🔧 App starting...")
    demo.launch(debug=True, share=False)