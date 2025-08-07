from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="demo.csv", source_column="source")

docs = loader.load()

print(docs)
print(len(docs))