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

template = "Write a {tone} email to {company} expressing interest in the {position}, mentioning {skill} as a key strength. Keep it to 4 lines max."

# Converting the template hwich the langchain can also understand.
from langchain_core.prompts import ChatPromptTemplate

""""
classmethod from_template(
template: str,
**kwargs: Any,
) → ChatPromptTemplate
Create a chat prompt template from a template string.

Creates a chat template consisting of a single message assumed to be from the human.

Parameters
:
template (str)  template string

**kwargs (Any)  keyword arguments to pass to the constructor.

Returns
:
A new instance of this class.
"""
prompt_template = ChatPromptTemplate.from_template(template)
# print(prompt_template,"\n\n")

prompt = prompt_template.invoke({
    "tone": "polite",
    "company": "Google",
    "position": "Software Engineer",
    "skill": "Python programming"
})

# print(prompt,"\n\n")

response = model.invoke(prompt)
# print(response.content)


# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system","You are a comedian who tells jokes about {topic}."),
    ("human","Tell me {joke_count} jokes"),
]

message_prompt_template = ChatPromptTemplate.from_messages(messages)
print(message_prompt_template,"\n\n")

message_prompt = message_prompt_template.invoke({ 
    "topic":"lawyers",
    "joke_count": 3
})
print(message_prompt,"\n\n")

"""
input_variables=['joke_count', 'topic'] input_types={} partial_variables={} messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], input_types={}, partial_variables={}, template='You are a comedian who tells jokes about {topic}.'), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['joke_count'], input_types={}, partial_variables={}, template='Tell me {joke_count} jokes'), additional_kwargs={})] 


messages=[SystemMessage(content='You are a comedian who tells jokes about lawyers.', additional_kwargs={}, response_metadata={}), HumanMessage(content='Tell me 3 jokes', additional_kwargs={}, response_metadata={})] 

"""
# Giving both the system message as well as the humna message.

response2 = model.invoke(message_prompt)
print(response2.content)
