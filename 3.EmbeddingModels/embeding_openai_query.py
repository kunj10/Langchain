from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 32)
text = "This is a test query."
query_result = embeddings.embed_query(text)
print(query_result)