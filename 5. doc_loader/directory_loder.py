from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loder = DirectoryLoader(path = 'book', glob = '*.pdf', loader_cls = PyPDFLoader)

docs = loder.load()

print(docs)
print(len(docs))