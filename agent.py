from openai import OpenAI
import requests
import os

class GaiaAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.instructions = (
            "You are a fact-based research assistant solving GAIA benchmark questions. "
            "For each question, reason step-by-step and output a single factual answer only."
        )

    def __call__(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
