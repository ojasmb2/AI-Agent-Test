# --- Standard Library Imports ---
import os
import requests
import pandas as pd
import gradio as gr
from typing import Union

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# You should modify this class to improve agent behavior.
class BasicAgent:
    def __init__(self):
        print("✅ BasicAgent initialized.")
    
    def __call__(self, question: str) -> str:
        # Print the incoming question (preview)
        print(f"🤖 Agent received question: {question[:50]}...")
        
        # Your logic goes here — modify as needed.
        fixed_answer = "This is a default answer."  # You can replace this with dynamic generation.
        
        print(f"📤 Agent returns: {fixed_answer}")
        return fixed_answer


# --- Core Evaluation Logic ---
def run_and_submit_all(profile: Union[gr.OAuthProfile, None]):
    """
    Core function that:
    - Initializes the agent
    - Fetches questions
    - Generates answers
    - Submits them to the scoring API
    - Returns the final result and answers DataFrame
    """
    space_id = os.getenv("SPACE_ID")  # Optional but used to link to the repo

    if profile:
        username = profile.username
        print(f"👤 Logged in user: {username}")
    else:
        print("⚠️ User not logged in.")
        return "Please login using Hugging Face Login button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "Not available"

    # 1. Instantiate the agent
    try:
        agent = BasicAgent()
    except Exception as e:
        return f"❌ Error initializing agent: {e}", None

    # 2. Fetch questions
    print(f"📥 Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "❌ Fetched questions list is empty or invalid.", None
        print(f"✅ {len(questions_data)} questions fetched.")
    except Exception as e:
        return f"❌ Error fetching questions: {e}", None

    # 3. Run the agent on all questions
    results_log = []
    answers_payload = []

    print("🧠 Running agent on questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}"
            })

    if not answers_payload:
        return "❌ Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }
    print(f"🚀 Submitting {len(answers_payload)} answers...")

    # 5. Submit answers to scoring endpoint
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()

        final_status = (
            f"✅ Submission Successful!\n"
            f"👤 User: {result_data.get('username')}\n"
            f"📊 Score: {result_data.get('score')}% "
            f"({result_data.get('correct_count')}/{result_data.get('total_attempted')} correct)\n"
            f"📝 Message: {result_data.get('message', 'No message.')}"
        )
        return final_status, pd.DataFrame(results_log)

    except requests.exceptions.HTTPError as e:
        return f"❌ Submission Failed (HTTP error): {e}", pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        return "❌ Submission Failed: Request timed out.", pd.DataFrame(results_log)
    except Exception as e:
        return f"❌ Submission Failed: {e}", pd.DataFrame(results_log)


# --- Gradio UI Setup ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Basic Agent Evaluation Tool")
    gr.Markdown("""
    ### 🛠 Instructions:
    1. Clone this Hugging Face Space.
    2. Implement your own logic in the `BasicAgent` class.
    3. Login with your Hugging Face account.
    4. Press the button to run all questions through your agent and submit.

    **Note:** It may take some time depending on the number of questions and agent logic.
    """)

    # Login and button interface
    gr.LoginButton()
    run_button = gr.Button("▶️ Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="📝 Submission Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="📄 Agent Answers Log", wrap=True)

    # Hook button click to function
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

# --- Local App Runner ---
if __name__ == "__main__":
    print("\n" + "-" * 30 + " 🚀 App Starting " + "-" * 30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")

    if space_host:
        print(f"✅ SPACE_HOST: {space_host}")
        print(f"🌍 App URL: https://{space_host}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not found (running locally?)")

    if space_id:
        print(f"✅ SPACE_ID: {space_id}")
        print(f"📦 Repo: https://huggingface.co/spaces/{space_id}")
    else:
        print("ℹ️  SPACE_ID not set")

    print("-" * 60)
    print("🔧 Launching Gradio app...")
    demo.launch(debug=True, share=False)
