from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv  
load_dotenv()   
llm = HuggingFaceEndpoint(model="zai-org/GLM-4.5")
model = ChatHuggingFace(llm=llm)   

prompt1 = PromptTemplate(
  template = "generate simple notes from following {text}",
  input_variables = ["text"] 
)

prompt2 = PromptTemplate(
  template = "generate questions for quiz from following {text}",
  input_variables = ["text"] 
)


prompt3 = PromptTemplate(
  template = "merge the provided notes with questions in single document, {notes} and {questions}",
  input_variables = ["notes", "questions"] 
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
  notes = prompt1 | model,
  questions = prompt2 | model
) | prompt3 | model | parser

result = parallel_chain.invoke({"text": "To understand general relativity, first, let's start with gravity, the force of attraction that two objects exert on one another. Sir Isaac Newton quantified gravity in the same text in which he formulated his three laws of motion, the Principial.The gravitational force tugging between two bodies depends on how massive each one is and how far apart the two lie, according to NASA Glenn Research Center. Even as the center of the Earth is pulling you toward it (keeping you firmly lodged on the ground), your center of mass is pulling back at the Earth. But the more massive body barely feels the tug from you, while with your much smaller mass, you find yourself firmly rooted thanks to that same force. Yet Newton's laws assume that gravity is an innate force of an object that can act over a distance.Albert Einstein, in his theory of special relativity, determined that the laws of physics are the same for all non-accelerating observers, and he showed that the speed of light within a vacuum is the same no matter the speed at which an observer travels, according to Wired.As a result, he found that space and time were interwoven into a single continuum known as space-time. And events that occur at the same time for one observer could occur at different times for another.Related: What would happen if the speed of light was much lower?As he worked out the equations for his general theory of relativity, Einstein realized that massive objects caused a distortion in space-time. Imagine setting a large object in the center of a trampoline. The object would press down into the fabric, causing it to dimple. If you then attempt to roll a marble around the edge of the trampoline, the marble would spiral inward toward the body, pulled in much the same way that the gravity of a planet pulls at rocks in space.In the decades since Einstein published his theories, scientists have observed countless of phenomena matching the predictions of relativity."})

print(result)