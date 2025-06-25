import gradio as gr
from agent import agent  # Import your refactored agent
from Gradio_UI import GradioUI

# Optional: Uncomment to test agent endpoint separately
# def run_agent(question):
#     try:
#         result = agent(question)
#         return [str(result)]
#     except Exception as e:
#         return [f"Error: {e}"]

if __name__ == "__main__":
    GradioUI(agent).launch(debug=True, share=True)