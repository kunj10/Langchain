from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)

class Output(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Output)

messages = "App is good in UI, but in performance it is slow. Also there is a chance of daa loss."

result = structured_model.invoke(messages)

print(result) 
print(result[summary])
print(result[sentiment])