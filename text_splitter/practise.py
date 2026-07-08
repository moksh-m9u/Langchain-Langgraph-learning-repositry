from langchain_community.document_loaders import (
    WebBaseLoader,
    PDFMinerLoader,
    UnstructuredPDFLoader  # <-- Changed from UnstructuredWordDocumentLoader
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

os.environ["UNSTRUCTURED_API_KEY"] = "LEqL6Q2ylCv0S1fHlM1KF25TlJ1lFO"

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed"
)

prompt = PromptTemplate(
    template="from this blog {blogs} answer the user with : \n {user_input}",
    input_variables=["blogs", "user_input"]
)

parser = StrOutputParser()

# Web loader
url = "https://mokshmannu.substack.com/p/data-version-control"
loader = WebBaseLoader(url)
docs = loader.load()

# PDF loader -- CORRECTED
pdf_loader = UnstructuredPDFLoader("datasheet.pdf")  # <-- Use this for PDFs
datasheet = pdf_loader.load()

print(datasheet)

chain = prompt | model | parser

while True:
    user = input("ask your question around my Blog: ")
    if user == "exit":
        break
    answer = chain.invoke({
        "blogs": datasheet,
        "user_input": user
    })
    print(f"here's your answer {answer}")