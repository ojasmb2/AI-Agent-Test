import os
import gradio as gr
import requests
import pandas as pd
from transformers import pipeline
from typing import Optional

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Smart Agent Definition ---
from transformers import pipeline

class BasicAgent:
    def __init__(self):
        print("Loading advanced model pipeline...")
        # You can swap this with another model if you want (like mistralai/Mistral-7B-Instruct-v0.2 if you use HF Inference API)
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

    def __call__(self, question: str) -> str:
        try:
            prompt = f"Answer the following question clearly and concisely:\n{question.strip()}"
            response = self.generator(prompt, max_new_tokens=128, do_sample=False, temperature=0.0)
            answer = response[0]["generated_text"].strip()
            return answer
        except Exception as e:
            print(f"Agent failed to answer question: {e}")
            return "ERROR"


# --- Submission Logic ---
def run_and_submit_all(profile: Optional[gr.OAuthProfile]):
    space_id = os.getenv("SPACE_ID")
    if not profile:
        print("User not logged in.")
        return "Please login to Hugging Face with the button.", None

    username = profile.username.strip()
    print(f"User logged in: {username}")

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent code link: {agent_code}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []

    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or not question_text:
            continue
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": answer})
        except Exception as e:
            error_msg = f"AGENT ERROR: {e}"
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": error_msg})

    if not answers_payload:
        return "No answers generated for submission.", pd.DataFrame(results_log)

    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }

    print(f"Submitting {len(answers_payload)} answers...")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"✅ Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')})\n"
            f"Message: {result_data.get('message', 'No message')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"❌ Submission failed: {e}", pd.DataFrame(results_log)

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Basic Agent Evaluation Runner")

    gr.Markdown(
        """
        **Instructions:**

        1. Clone this space and implement your agent logic.
        2. Log in with your Hugging Face account using the button below.
        3. Click **Run Evaluation & Submit All Answers** to test and submit your agent.

        ---
        ⚠️ Note: The first run may take time depending on model and question count.
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

# --- Run App ---
if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST: {space_host_startup}")
        print(f"Runtime URL: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST not set.")

    if space_id_startup:
        print(f"✅ SPACE_ID: {space_id_startup}")
        print(f"Repo: https://huggingface.co/spaces/{space_id_startup}")
    else:
        print("ℹ️ SPACE_ID not set.")

    print("-" * 80)
    print("Launching Gradio App...")
    demo.launch(debug=True, share=False)
