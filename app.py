import os
import gradio as gr
import requests
import inspect
import pandas as pd

# --- Hugging Face Agents & Tools imports ---
from transformers import load_tool, ReactAgent

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Load Tools ---
# Document QA tool
qa_tool = load_tool(
    task_or_repo_id="document_question_answering",
    model_repo_id="deepset/roberta-base-squad2"
)
# Web search tool
web_tool = load_tool(
    task_or_repo_id="search"
)
# Python REPL tool
python_tool = load_tool(
    task_or_repo_id="python_repl"
)

# --- Agent Definition ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized with real tools.")
        # Initialize a ReAct agent with the loaded tools
        self.agent = ReactAgent(
            tools=[qa_tool, web_tool, python_tool],
            llm_engine="openai/chat:gpt-3.5-turbo",
            verbose=True
        )

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        try:
            answer = self.agent.run(question)
            print(f"Agent returning answer: {answer}")
            return answer
        except Exception as e:
            print(f"Error in agent execution: {e}")
            return f"AGENT ERROR: {e}"

# --- Evaluation & Submission Logic ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code at: {agent_code}")

    # 2. Fetch Questions
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    # 3. Run Agent on each question
    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        submitted_answer = agent(question_text)
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Submitted Answer": submitted_answer
        })

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Submit Answers
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except Exception as e:
        print(f"Submission error: {e}")
        results_df = pd.DataFrame(results_log)
        return f"Submission Failed: {e}", results_df

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1. Clone this space and modify the code to define your agent's logic and tools.
        2. Log in with Hugging Face to submit under your username.
        3. Click 'Run Evaluation & Submit All Answers' to fetch questions, run the agent, and submit.
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(debug=True, share=False)
