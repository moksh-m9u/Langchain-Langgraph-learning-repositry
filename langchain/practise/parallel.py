from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=None)
model2 = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=None)

prompt1 = PromptTemplate(
    template="Generate simple and detailed notes from the following texts \n {texts}",
    input_variables=["texts"]
)
prompt2 = PromptTemplate(
    template="Generate 5 tricky question answer from following texts \n {texts}",
    input_variables=["texts"]
)
prompt3 = ChatPromptTemplate.from_template(
    "Merge the provided quiz and notes into a single coherent response \n Notes: {notes} \n Quiz: {quiz}"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

merge_chain = prompt3 | model1 | parser

full_chain = parallel_chain | merge_chain

result = full_chain.invoke({"texts": "Machine learning is a subset of AI that enables systems to learn from data."})
print(result)
