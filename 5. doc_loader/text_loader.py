from  langchain.document_loaders import TextLoader

loader = TextLoader('demo.txt')

docs = loader.load()

print(docs)
print(len(docs))