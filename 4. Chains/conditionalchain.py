from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv  
load_dotenv()   

# Load HF model
llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)   

# Pydantic class to parse sentiment
class Feedback(BaseModel):
  sentiment: Literal["positive", "negative"] = Field(..., description="Sentiment of the feedback")

# Output parser for sentiment only
parser = PydanticOutputParser(pydantic_object=Feedback)

# Sentiment classification prompt
prompt1 = PromptTemplate(
  template = "Classify the sentiment of the following feedback as positive or negative:\n\n{feedback}\n\n{format_instructions}",
  input_variables = ["feedback"],
  partial_variables = {"format_instructions": parser.get_format_instructions()}
)

# Response generation prompts
positive_prompt = PromptTemplate(
  template = "Write a thank-you message to the customer for their positive feedback:\n\n{feedback}",
  input_variables = ["feedback"] 
)

negative_prompt = PromptTemplate(
   template = "Write a message to the customer acknowledging their negative feedback and assuring improvement:\n\n{feedback}",
   input_variables = ["feedback"] 
)

# Sentiment classification chain
sentiment_chain = prompt1 | model | parser

# ✅ Use StrOutputParser for the second stage
text_parser = StrOutputParser()

# Branch chain: select message type based on sentiment
branch_chain = RunnableBranch(
  (lambda x: x.sentiment == "positive", positive_prompt | model | text_parser),
  (lambda x: x.sentiment == "negative", negative_prompt | model | text_parser),
  RunnableLambda(lambda _: "No sentiment detected.")
)

# Final chain
final_chain = sentiment_chain | branch_chain

# Test input
result = final_chain.invoke({"feedback": "I love this product"})
print(result)
