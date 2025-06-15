import json
import os
from typing import List, Dict
from agent import GAIAAgent

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    
    # Remove common prefixes/suffixes
    answer = answer.strip()
    
    # Remove quotes if they wrap the entire answer
    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Convert to lowercase for comparison
    return answer.lower().strip()

def extract_final_answer(response: str) -> str:
    """Extract the final answer from the model response."""
    if "FINAL ANSWER:" in response:
        answer = response.split("FINAL ANSWER:")[1].strip()
        # Clean up the answer - remove any trailing explanation
        answer = answer.split('\n')[0].strip()
        return answer
    
    # If no FINAL ANSWER format, try to extract from end of response
    lines = response.strip().split('\n')
    return lines[-1].strip()

def load_gaia_dataset(dataset_path: str) -> List[Dict]:
    """Load GAIA dataset from JSON/JSONL file."""
    tasks = []
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return tasks
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            if dataset_path.endswith('.jsonl'):
                # JSONL format - one JSON object per line
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            task = json.loads(line)
                            tasks.append(task)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
            else:
                # Regular JSON format
                data = json.load(f)
                if isinstance(data, list):
                    tasks = data
                elif isinstance(data, dict) and 'tasks' in data:
                    tasks = data['tasks']
                else:
                    print("Unexpected JSON format")
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    print(f"Loaded {len(tasks)} tasks from {dataset_path}")
    return tasks

def create_sample_dataset() -> List[Dict]:
    """Create a sample dataset for testing if no GAIA dataset is available."""
    sample_tasks = [
        {
            "task_id": "sample_1",
            "question": "What is 15 + 27?",
            "answer": "42",
            "level": 1,
            "file_name": None
        },
        {
            "task_id": "sample_2", 
            "question": "What is the capital of France?",
            "answer": "Paris",
            "level": 1,
            "file_name": None
        },
        {
            "task_id": "sample_3",
            "question": "How many days are in a leap year?",
            "answer": "366",
            "level": 1,
            "file_name": None
        },
        {
            "task_id": "sample_4",
            "question": "What is 2 * 6 * 7?",
            "answer": "84",
            "level": 1,
            "file_name": None
        },
        {
            "task_id": "sample_5",
            "question": "What year did World War II end?",
            "answer": "1945",
            "level": 1,
            "file_name": None
        }
    ]
    
    print("Using sample dataset for testing")
    return sample_tasks

def evaluate_agent(dataset_path: str = None, max_tasks: int = None) -> float:
    """Evaluate the GAIA agent on the dataset."""
    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        tasks = load_gaia_dataset(dataset_path)
    else:
        print("No dataset file found, using sample tasks for testing")
        tasks = create_sample_dataset()
    
    if not tasks:
        print("No tasks to evaluate")
        return 0.0
    
    # Limit number of tasks if specified
    if max_tasks:
        tasks = tasks[:max_tasks]
        print(f"Evaluating on first {len(tasks)} tasks")
    
    # Initialize agent
    print("Initializing GAIA agent...")
    agent = GAIAAgent()
    
    # Test API connection first
    print("Testing API connection...")
    test_response = agent.test_grok()
    if "error" in test_response.lower():
        print(f"API test failed: {test_response}")
        return 0.0
    else:
        print("API connection successful!")
    
    # Process tasks
    correct = 0
    total = len(tasks)
    submission_entries = []
    
    print(f"\nStarting evaluation on {total} tasks...")
    print("=" * 50)
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", f"task_{i}")
        question = task.get("question", "")
        expected_answer = task.get("answer", "")
        
        print(f"\nTask {i}/{total}: {task_id}")
        print(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Process task with agent
            response = agent.process_task(task)
            predicted_answer = extract_final_answer(response)
            
            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            
            # Compare answers (normalized)
            is_correct = normalize_answer(predicted_answer) == normalize_answer(expected_answer)
            
            if is_correct:
                correct += 1
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
            
            # Store submission entry
            submission_entries.append({
                "task_id": task_id,
                "model_answer": predicted_answer,
                "reasoning_trace": response
            })
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            submission_entries.append({
                "task_id": task_id,
                "model_answer": "ERROR",
                "reasoning_trace": f"Error: {str(e)}"
            })
        
        # Progress update
        current_score = (correct / i) * 100
        print(f"Current score: {correct}/{i} = {current_score:.1f}%")
        print("-" * 30)
    
    # Final score
    final_score = (correct / total) * 100
    
    # Save submission file
    try:
        with open("submission.jsonl", "w", encoding="utf-8") as f:
            for entry in submission_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"\nSubmission saved to submission.jsonl")
    except Exception as e:
        print(f"Error saving submission: {e}")
    
    # Print final results
    print("=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Total tasks: {total}")
    print(f"Correct answers: {correct}")
    print(f"Final score: {final_score:.2f}%")
    
    if final_score >= 30:
        print("🎉 CONGRATULATIONS! Score ≥30% - Certificate achieved!")
    else:
        print(f"📈 Score below 30%. Need {30 - final_score:.2f}% more for certificate.")
    
    return final_score

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GAIA agent")
    parser.add_argument("--dataset", type=str, default="gaia_test.json", 
                       help="Path to GAIA dataset file")
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Maximum number of tasks to evaluate")
    
    args = parser.parse_args()
    
    score = evaluate_agent(args.dataset, args.max_tasks)
    
    print(f"\nFinal evaluation score: {score:.2f}%")
    
    if score >= 30:
        print("Certificate requirements met! 🎉")
    else:
        print("Keep working to reach 30% for the certificate! 💪")

if __name__ == "__main__":
    main() 