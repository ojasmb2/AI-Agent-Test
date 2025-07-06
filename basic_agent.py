from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class AgentState(TypedDict):
    question: str
    answer: str
    task_id: str
    log: Annotated[List[str], operator.add]

def assistant(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: <your answer here>. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(messages)
    return {"answer": response.content, "log": [f"Assistant response: {response.content}"]}

# Functions to interact with the API
def get_all_questions():
    """Fetch all questions from the API"""
    try:
        response = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return []

def get_random_question():
    """Fetch a random question from the API"""
    try:
        response = requests.get(f"{DEFAULT_API_URL}/random-question", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching random question: {e}")
        return None

def get_file_for_task(task_id: str):
    """Download file associated with a task ID"""
    try:
        response = requests.get(f"{DEFAULT_API_URL}/files/{task_id}", timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file for task {task_id}: {e}")
        return None

def submit_answers(username: str, agent_code: str, answers: List[Dict]):
    """Submit answers to the API"""
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers
    }
    try:
        response = requests.post(f"{DEFAULT_API_URL}/submit", json=submission_data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error submitting answers: {e}")
        return None

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("assistant", assistant)
graph.add_edge(START, "assistant")
graph.add_edge("assistant", END)
app = graph.compile()

def run_agent(question: str, task_id: str):
    """Run the agent on a single question"""
    state = {"question": question, "task_id": task_id, "log": []}
    return app.invoke(state)

def run_agent_on_all_questions():
    """Run the agent on all questions from the API"""
    print("Fetching all questions...")
    questions = get_all_questions()
    
    if not questions:
        print("No questions found or error occurred")
        return
    
    print(f"Found {len(questions)} questions")
    results = []
    
    for i, question_data in enumerate(questions):
        task_id = question_data.get("task_id")
        question_text = question_data.get("question")
        
        if not task_id or not question_text:
            print(f"Skipping malformed question {i}")
            continue
        
        print(f"\nProcessing question {i+1}/{len(questions)}")
        print(f"Task ID: {task_id}")
        print(f"Question: {question_text[:100]}...")
        
        # Run the agent
        result = run_agent(question_text, task_id)
        
        results.append({
            "task_id": task_id,
            "question": question_text,
            "answer": result["answer"],
            "log": result["log"]
        })
        
        print(f"Answer: {result['answer']}")
    
    return results

def demo_single_question():
    """Demo with a single random question"""
    print("Fetching a random question...")
    question_data = get_random_question()
    
    if not question_data:
        print("Could not fetch random question")
        return
    
    task_id = question_data.get("task_id")
    question_text = question_data.get("question")
    
    print(f"Task ID: {task_id}")
    print(f"Question: {question_text}")
    
    # Run the agent
    result = run_agent(question_text, task_id)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Log: {result['log']}")
    
    return result

if __name__ == "__main__":
    # Option 1: Test with a single random question
    # print("=== Testing with Random Question ===")
    # demo_single_question()
    
    # print("\n" + "="*50 + "\n")
    
    # Option 2: Run on all questions (commented out for now)
    print("=== Running on All Questions ===")
    results = run_agent_on_all_questions()
    
    # Save results to file
    if results:
        with open('agent_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to agent_results.json")
    
    # Option 3: Manual question for testing
    print("=== Manual Test ===")
    manual_question = "What is the capital of France?"
    manual_task_id = "test-123"
    manual_result = run_agent(manual_question, manual_task_id)
    print(f"Question: {manual_question}")
    print(f"Answer: {manual_result['answer']}")