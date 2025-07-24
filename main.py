from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS so Streamlit can access it if hosted elsewhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_agent(request: Request):
    body = await request.json()
    question = body.get("question", "")

    # Call LangGraph agent here
    # response = agent.invoke(...) or whatever your method is
    response = {"response": f"Simulated answer to: {question}"}  # Replace this

    return response