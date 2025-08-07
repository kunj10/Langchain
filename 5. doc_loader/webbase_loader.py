from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.google.com")

docs = loader.load()

print(docs)
print(len(docs))