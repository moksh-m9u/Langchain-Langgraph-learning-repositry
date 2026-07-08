from langchain_community.document_loaders import TextLoader , PyPDFLoader , PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.8,
    reasoning_format="hidden"
)

prompt = PromptTemplate(
    template="write a summary for the following {blog}",
    input_variables=['blog']
)

parser = StrOutputParser()
loader = PyPDFLoader("datasheet.pdf")

docs = loader.load()
print(docs[2].page_content)
#print(docs[0].metadata)
"""
chain= prompt | model | parser 
result = chain.invoke({"blog":docs[0]})
print (result)

"""