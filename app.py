import os
import gradio as gr
import requests
import inspect
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Advanced GAIA-Ready Agent ---
class GaiaAgent:
    def __init__(self):
        print("Initializing GaiaAgent with open-source model...")

        model_name = "google/flan-t5-large"  # Good balance between size and reasoning quality
        auth_token = os.getenv("HF_TOKEN")

        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            token=auth_token,
            device=self.device
        )
        print("Model and tokenizer loaded.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question: {question[:60]}...")
        prompt = (
            f"Answer the following question as accurately as possible.\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        try:
            result = self.pipe(prompt, max_new_tokens=64, clean_up_tokenization_spaces=True)[0]["generated_text"]
            # Ensure clean return without "Answer:" prefix
            answer = result.strip().replace("Answer:", "").strip()
            print(f"Agent returned: {answer}")
            return answer
        except Exception as e:
            print(f"Error during model inference: {e}")
            return f"AGENT ERROR: {e}"

# --- Evaluation & Submission Logic ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GaiaAgent()
    except Exception as e:
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        return f"Error decoding server response for questions: {e}", None

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
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")

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
    except requests.exceptions.RequestException as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)
    except Exception as e:
        return f"An unexpected error occurred during submission: {e}", pd.DataFrame(results_log)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA-Level Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Modify and extend the agent in the code section.
        2. Login with your Hugging Face account to submit answers.
        3. Click the button to run and submit.

        ---
        *This agent uses `google/flan-t5-large` from Hugging Face to answer questions.*
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
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST not found.")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
    else:
        print("ℹ️  SPACE_ID not found.")

    print("-"*(60 + len(" App Starting ")) + "\n")
    demo.launch(debug=True, share=False)
