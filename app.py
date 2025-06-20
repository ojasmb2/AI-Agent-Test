from transformers import pipeline

class BasicAgent:
    def __init__(self):
        print("🔄 Loading Meta-Llama-3-8B-Instruct model...")
        self.pipeline = pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_length=300,
            temperature=0.2,
            do_sample=False
        )

        self.few_shot_examples = """
Q: What is the capital of Germany?
A: Berlin

Q: Who discovered gravity?
A: Isaac Newton

Q: What is 5 multiplied by 9?
A: 45

Q: What is the smallest prime number?
A: 2
"""

    def __call__(self, question: str) -> str:
        try:
            prompt = (
                "You are a helpful and intelligent agent. Answer concisely.\n\n"
                + self.few_shot_examples
                + f"\nQ: {question}\nA:"
            )
            response = self.pipeline(prompt)[0]["generated_text"]
            answer = response.split("A:")[-1].strip()
            return answer.split("\n")[0].strip()
        except Exception as e:
            print("Error:", e)
            return "ERROR"
