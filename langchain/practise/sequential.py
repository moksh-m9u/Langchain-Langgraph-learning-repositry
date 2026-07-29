from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b",temperature=0,max_tokens=None)

prompt = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following {text}",
    input_variables=['text']
)
 
parser= StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser
result = chain.invoke({"topic":"dvc"})
result2 = chain.invoke({"topic":"dvc"})
print (result)