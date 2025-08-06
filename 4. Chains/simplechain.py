from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  
load_dotenv()   
llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)   

prompt = PromptTemplate(
  template = "generate 5 facts about {topic}",
  input_variables = ["topic"] 
)

parser = StrOutputParser()

chian = prompt | model | parser

result = chian.invoke({"topic": "India"})

print(result)