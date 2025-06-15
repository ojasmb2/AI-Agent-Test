import gradio as gr
import json
import os
from datetime import datetime
from agent import GAIAAgent
from evaluate import evaluate_agent, create_sample_dataset
import traceback

def run_evaluation():
    """Run the GAIA evaluation and return results."""
    try:
        print("Starting GAIA Agent Evaluation...")
        print("=" * 50)
        
        # Initialize agent
        agent = GAIAAgent()
        
        # Test API connection first
        print("Testing xAI API connection...")
        test_response = agent.test_grok()
        print(f"API Test Response: {test_response}")
        
        # Run evaluation on sample dataset (since we don't have the full GAIA dataset)
        print("\nRunning evaluation on sample tasks...")
        score = evaluate_agent(dataset_path=None, max_tasks=10)
        
        # Read submission file if it exists
        submission_content = ""
        if os.path.exists("submission.jsonl"):
            with open("submission.jsonl", "r") as f:
                submission_content = f.read()
        
        # Format results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = f"""
# GAIA Agent Evaluation Results

**Timestamp:** {timestamp}
**Final Score:** {score:.2f}%
**Certificate Status:** {'✅ ACHIEVED (≥30%)' if score >= 30 else '❌ NOT ACHIEVED (<30%)'}

## API Connection Status
{test_response}

## Submission File Preview
```json
{submission_content[:500]}{'...' if len(submission_content) > 500 else ''}
```

## Next Steps
{'🎉 Congratulations! You can now claim your Certificate of Excellence!' if score >= 30 else '💪 Keep improving your agent to reach the 30% threshold.'}
        """
        
        return results, score
        
    except Exception as e:
        error_msg = f"""
# Evaluation Error

**Error:** {str(e)}

**Traceback:**
```
{traceback.format_exc()}
```

Please check the logs and fix any issues before retrying.
        """
        return error_msg, 0.0

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="GAIA Agent Evaluation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 GAIA Agent Evaluation
        
        This is your GAIA benchmark agent for the Hugging Face Agents Course Certificate of Excellence.
        
        **Goal:** Achieve ≥30% score on GAIA benchmark tasks
        
        Click the button below to run the evaluation and submit your answers.
        
        ⚠️ **Note:** This may take several minutes to complete. Please be patient.
        """)
        
        with gr.Row():
            run_btn = gr.Button("🚀 Run Evaluation & Submit All Answers", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Run Status / Submission Result")
                results_output = gr.Markdown("Click the button above to start evaluation...")
            
            with gr.Column():
                gr.Markdown("## Score")
                score_output = gr.Number(label="Final Score (%)", value=0.0, interactive=False)
        
        # Event handler
        run_btn.click(
            fn=run_evaluation,
            inputs=[],
            outputs=[results_output, score_output],
            show_progress=True
        )
        
        gr.Markdown("""
        ---
        
        ## About This Agent
        
        - **API:** xAI Grok for reasoning
        - **Tools:** Web search, file handling, math calculations
        - **Fallbacks:** Local knowledge for common questions
        - **Target:** 30% accuracy for certificate eligibility
        
        ## Troubleshooting
        
        If you encounter issues:
        1. Check the container logs in the "Logs" tab
        2. Verify API credentials and internet connectivity
        3. Ensure all dependencies are installed
        
        **Good luck! 🍀**
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=False
    ) 