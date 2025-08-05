from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatAnthropic(model="claude-3-7-sonnet-20250219")
"""
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, world!"}
]
"""
result = chat_model.invoke("PM of India")

print(result.content)