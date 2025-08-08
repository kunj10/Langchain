from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
load_dotenv()   
llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)   

retriever = WikipediaRetriever(top_k=2)

query = "The geopolitical shift of US and India ?"

docs = retriever.invoke(query)

print(docs)
