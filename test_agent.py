#!/usr/bin/env python3
"""
Test script to verify GAIA agent setup and functionality.
"""

from agent import GAIAAgent
from tools import web_search, read_file, calculate_simple_math

def test_api_connection():
    """Test xAI API connection."""
    print("Testing xAI API connection...")
    agent = GAIAAgent()
    
    try:
        response = agent.test_grok()
        print(f"API Response: {response}")
        
        if "error" in response.lower():
            print("❌ API test failed")
            return False
        else:
            print("✅ API connection successful")
            return True
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def test_basic_reasoning():
    """Test basic reasoning capabilities."""
    print("\nTesting basic reasoning...")
    agent = GAIAAgent()
    
    test_cases = [
        {
            "task_id": "test_math",
            "question": "What is 25 + 17?", 
            "expected": "42"
        },
        {
            "task_id": "test_general",
            "question": "What is the capital of Japan?",
            "expected": "tokyo"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['question']}")
        try:
            response = agent.process_task(test_case)
            predicted = agent.extract_final_answer(response)
            print(f"Response: {predicted}")
            
            # Simple comparison
            if test_case['expected'].lower() in predicted.lower():
                print("✅ Test passed")
            else:
                print("❌ Test failed")
                
        except Exception as e:
            print(f"❌ Test error: {e}")

def test_tools():
    """Test individual tools."""
    print("\nTesting tools...")
    
    # Test math calculation
    print("\n1. Testing math calculation:")
    result = calculate_simple_math("15 + 27")
    print(f"15 + 27 = {result}")
    
    # Test web search (fallback)
    print("\n2. Testing web search:")
    search_result = web_search("capital of France", None)
    print(f"Search result: {search_result[:100]}...")
    
    # Test file reading (with non-existent file)
    print("\n3. Testing file reading:")
    file_result = read_file("nonexistent.txt")
    print(f"File read result: {file_result}")

def test_sample_task():
    """Test with a sample GAIA-like task."""
    print("\nTesting sample GAIA task...")
    
    agent = GAIAAgent()
    
    sample_task = {
        "task_id": "sample_test",
        "question": "If a store has 150 apples and sells 87 of them, how many apples are left?",
        "answer": "63",
        "file_name": None
    }
    
    try:
        print(f"Question: {sample_task['question']}")
        response = agent.process_task(sample_task)
        predicted = agent.extract_final_answer(response)
        expected = sample_task['answer']
        
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        
        if predicted.strip() == expected:
            print("✅ Sample task passed")
        else:
            print("❌ Sample task failed")
            
    except Exception as e:
        print(f"❌ Sample task error: {e}")

def main():
    """Run all tests."""
    print("GAIA Agent Test Suite")
    print("=" * 50)
    
    # Test API connection first
    api_ok = test_api_connection()
    
    if not api_ok:
        print("\n❌ API connection failed. Cannot proceed with other tests.")
        print("Please check your API key and internet connection.")
        return
    
    # Run other tests
    test_basic_reasoning()
    test_tools()
    test_sample_task()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("If all tests passed, you can run: python evaluate.py")

if __name__ == "__main__":
    main() 