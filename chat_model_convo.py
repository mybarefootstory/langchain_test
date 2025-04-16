from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2
)

messages = [
    SystemMessage(
        content="You are an expert in Social Media Content Strategy"
    ),
    HumanMessage(content="My name is Akash and give me a short tip on creating engaging posts on Instagram."),
]

result = llm.invoke(messages)
print(result.content,"\n")
print(result.response_metadata,"\n")