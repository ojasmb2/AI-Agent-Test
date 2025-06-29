from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class BasicAgent:
    def __init__(self):
        print("Loading model...")
        model_id = "google/flan-t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("BasicAgent with FLAN-T5 initialized.")

    def __call__(self, question: str) -> str:
        prompt = (
            "You are a general AI assistant. I will ask you a question. "
            "Report your thoughts, and finish your answer with the following template: "
            "FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible "
            "OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number "
            "neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, "
            "neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. "
            f"Now, here is the question: {question}"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Answer: {answer}")
        return answer