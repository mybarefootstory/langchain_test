from dotenv import load_dotenv

import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# print(GROQ_API_KEY)

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "My name is Akash, What is your name."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content,"\n")
print(ai_msg.response_metadata,"\n")
print(ai_msg.response_metadata["model_name"])