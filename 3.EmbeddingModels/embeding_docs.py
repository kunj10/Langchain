from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv() 

llm = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction")


docs = ["what is the capital of India?",
        "what is the capital of USA?",
        "what is the capital of Australia?"]

result = llm.embed_documents(docs)

print(result)