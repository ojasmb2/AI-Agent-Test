from llama_index.llms.openai import OpenAI
from llama_index.tools.wikipedia.base import WikipediaToolSpec
from llama_index.core.llms import ChatMessage
from llama_index.core.agent import ReActAgent
import logging
from llama_index.llms.deepinfra import DeepInfraLLM
import os
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.prompts import PromptTemplate
import requests
import json

class CuongBasicAgent:
    """
    Agent using LlamaIndex to fetch data from the web and answer GAIA benchmark questions.
    """
    def __init__(self):
        system_prompt = """
        Value: You are an advanced assistant designed to help with a variety of tasks, including answering questions, providing summaries, and performing other types of analyses.

        ## Tools

        You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
        This may require breaking the task into subtasks and using different tools to complete each subtask.

        You have access to the following tools:
        {tool_desc}


        ## Output Format

        Please answer in the same language as the question and use the following format:

        ```
        Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
        Action: tool name (one of {tool_names}) if using a tool.
        Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
        ```

        Please ALWAYS start with a Thought.

        NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

        Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

        If this format is used, the tool will respond in the following format:

        ```
        Observation: tool response
        ```

        You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

        ```
        Thought: I can answer without using any more tools. I'll use the user's language to answer
        Answer: [your answer here (In the same language as the user's question)]
        ```

        ```
        Thought: I cannot answer the question with the provided tools.
        Answer: [your answer here (In the same language as the user's question)]
        ```

        The answer should be concise and to the point. For example, if the answer is a number, just return the number without any additional text. If the question is "What is the capital of France?", the answer should be "Paris".

        If the question includes guidelines regarding the format of the answer, please follow those guidelines faithfully
        ## Current Conversation

        Below is the current conversation consisting of interleaving human and assistant messages.
        """
        react_system_prompt = PromptTemplate(system_prompt)
        #llm = DeepInfraLLM(
        #    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",api_key=os.getenv("DEEPINFRA_API_KEY"))

        llm = OpenAI(model='gpt-4.1')
        agent = ReActAgent.from_tools(
            llm=llm,
            tools=WikipediaToolSpec().to_tool_list() + TavilyToolSpec(api_key=os.getenv('TAVILY_API_KEY')).to_tool_list(),
            verbose=True,
        )

        agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
        self.agent = agent

    def __call__(self, question: str) -> str:
        answer = self.agent.query(question)
        return str(answer)