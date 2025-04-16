from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2
)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI Assistant.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")




