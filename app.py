import os
import gradio as gr
from agent import build_agent_executor
from huggingface_hub import InferenceClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN is missing. Please set it as an environment variable.")

# Initialize LLM client
llm_client = InferenceClient(token=HF_TOKEN)

# Build the agent
agent_executor = build_agent_executor(llm_client)

def run_agent(user_input: str) -> str:
    try:
        logger.info("User input: %s", user_input)
        response = agent_executor.run(user_input)
        logger.info("Agent response: %s", response)
        return response
    except Exception as e:
        logger.error("Error during agent execution: %s", e, exc_info=True)
        return f"Something went wrong: {e}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Professional HF Agent\nEnter a query and watch the agent in action.")
    
    with gr.Row():
        user_input = gr.Textbox(label="Your Question", placeholder="Ask me anything...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
        output_box = gr.Textbox(label="Agent Response", lines=6)

    submit_btn.click(fn=run_agent, inputs=user_input, outputs=output_box)

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()
