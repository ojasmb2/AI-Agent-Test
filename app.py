import os
import gradio as gr
import requests
import pandas as pd

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space/questions"

# --- Basic Agent Definition ---
class MyAgent:
    def __init__(self):
        print("MyAgent initialized.")
    
    def __call__(self, question: str) -> str:
        print(f"Received question: {question}")
        # 這裡返回簡單的動態回答，根據問題的內容
        return f"Answer to '{question[:30]}...'"  # 簡單返回問題的片段

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- 確認 Hugging Face 使用者是否已登入 ---
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    # --- 定義 API 端點 ---
    api_url = DEFAULT_API_URL
    questions_url = api_url  # 這裡直接用 api_url，無需再加 '/questions'
    submit_url = f"{api_url}/submit"

    # 1. 初始化代理 (Agent)
    try:
        agent = MyAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    # 2. 抓取問題
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)  # 直接用 api_url
        response.raise_for_status()
        questions_data = response.json()

        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None

        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. 執行代理 (Agent)
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        
        try:
            # 在這裡運行代理並獲得答案
            submitted_answer = agent(question_text)  # 假設 `agent` 能處理這個問題並返回答案
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    # 檢查是否有答案可以提交
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. 準備提交
    submission_data = {"answers": answers_payload}
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()  # This will raise an exception for 4xx or 5xx responses
        result_data = response.json()

        # 提取回應中的相關信息
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        
        # 轉換為 DataFrame 以方便檢查
        results_df = pd.DataFrame(results_log)
        return final_status, results_df

    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()  # Hugging Face Login Button

    # Optionally, allow user to choose if they want to submit answers or just run evaluation
    eval_mode = gr.Checkbox(label="Run Evaluation Only", value=True)

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    # Modify the function call to include the mode
    run_button.click(
        fn=run_and_submit_all,
        inputs=[eval_mode],  # Pass the evaluation mode as an input
        outputs=[status_output, results_table]
    )

# Keep the startup information part unchanged
if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)