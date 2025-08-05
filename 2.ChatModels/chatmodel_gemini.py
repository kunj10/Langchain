from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, world!"}
]
"""
result = chat_model.invoke("PM of India")

print(result.content)