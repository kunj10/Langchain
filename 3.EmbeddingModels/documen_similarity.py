from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from dotenv import load_dotenv
load_dotenv() 


llm = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-mpnet-base-v2",
    task="feature-extraction")

docs = ["My name is Kunj",
        "I am learing GENAI",
        "I am live in India",
        "My Favourite color is Black"] 

query = "from where are you?"

doc_embedding  = llm.embed_documents(docs)
query_embedding = llm.embed_query(query)

similarity_matrix = cosine_similarity([query_embedding], doc_embedding)[0]


index, score = sorted(list(enumerate(similarity_matrix)), key=lambda x: x[1], reverse=True)[0]

print(query)

print(docs[index])

print("similarity score: ", score)
