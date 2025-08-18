---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---
Directions are for a mac system with apple silicone:
To locally run the agent on your local system, make sure to pip install all the required files. Complete list is not created yet but keep running in terminal until all required files are satisfied. This model runs gpt 4o so make sure to include api key when using.

Required installations:
pip install python-dotenv
pip install requests
pip install langgraph
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install langchain-groq
pip install langchain-huggingface
pip install langchain-ollama
pip install supabase
pip install sentence-transformers

First step is to start the mcp server using...


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference