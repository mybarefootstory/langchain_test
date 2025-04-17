from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence, RunnableParallel
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

# Define prompt template for the movei summary
Summary_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie expert who summarizes movies"),
        ("human","Summarize the movie {movie}"),
    ]
)

# Define plot analysis step
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic."),
            ("human","Analyze the plot: {plot}. What are its strengths and weaknesses?")

        ]
    )
    return plot_template.format_prompt(plot=plot)

# Define character analysis step
def analyze_characters(plot):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic."),
            ("human","Analyze the characters: {characters}. What are their strengths and weaknesses?")
        ]
    )
    return character_template.format_prompt(characters=plot)

# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"


# Simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | model | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    Summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches = {"plot": plot_branch_chain, "charcter": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["charcter"]))
)

# Run the chain
result = chain.invoke({"movie":"Intersteller"})
print(result)






