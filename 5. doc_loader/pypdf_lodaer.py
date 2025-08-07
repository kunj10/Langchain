from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('5. doc_loader/RAG.pdf')

docs = loader.load()

print(docs)
print(len(docs))