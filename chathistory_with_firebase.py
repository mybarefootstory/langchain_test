from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq

"""
1. Create a Firebase account
2. Create a new Firebase account
    - Copy the Project ID
3. Create a firebase database in the Firebase Project
4. Install teh Google Cloud CLI on your computer
5. pip install google-cloud-firestore
6. Enable the Firestore API in the Google Cloud Console.
"""

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchain-practise-15d09"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client....")
client = firestore.Client(project=PROJECT_ID)

# Initialize FireStore Chat Message History
print("Initializing Firestore Chat Message History....")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME, 
    client = client,
)

print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2
)

print("Starting chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
