from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv  
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",
    task="text-generation")

chat_model = ChatHuggingFace(llm=llm)

load_dotenv()   

model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")