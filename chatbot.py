from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Define the endpoint-based model correctly
llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)

chat_history = [
  SystemMessage(content="You are a helpful assistant")
]

while True:
  user_input = input("User: ")
  chat_history.append(HumanMessage(content=user_input))
  if user_input == "exit":
    break
  result = model.invoke(chat_history)
  chat_history.append(AIMessage(content=result.content))
  print("Bot: ", result.content)

print("Chat History: ", chat_history)


