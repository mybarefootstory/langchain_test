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

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a facts expert who knows about {animal}"),
        ("human","Tell me {fact_count} facts.")
    ]
)

# Create a combined chain using Langchain Expression Language (LCEL)
# StrOutputParser, no need to pull .content everytime, it's done auto
chain = prompt_template | model | StrOutputParser()
print(chain,"\n\n")

# Run the chain
result = chain.invoke({"animal":"cat","fact_count":5})

print(result)
