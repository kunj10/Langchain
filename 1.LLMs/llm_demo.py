from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm= OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

result = llm.invoke("Hello, world!")

print(result)
