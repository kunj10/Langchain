from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI(model="gpt-4")
"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, world!"}
]
"""
result = chat_model.invoke("PM of India")

print(result.content)