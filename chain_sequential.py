from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Create a ChatGroq model Instance
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    timeout=None,
    max_retries=2
)

# Define prompt Templates
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system","You love facts and you tell short facts about {animal}"),
        ("human","Tell me {count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system","Translate the following text to {language} "),
        ("human","{text}"),
    ]
)

# Define additional Processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {
    "text":output,
    "language":"Hindi"
})

# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

# Run the chain
result = chain.invoke({"animal":"cat","count":5})
print(result)

