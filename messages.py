from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from lanchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv() 

llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello, how are you doing?")]

result = model.invoke(messages)


messages.append(AIMessage(content=result.content))


print(messages)