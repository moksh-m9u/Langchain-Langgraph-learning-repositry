from langchain_community.tools import tool
from langchain_community.tools import ShellTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
load_dotenv()

'''
search_tool = DuckDuckGoSearchRun()
print(search_tool.invoke("details about Moksh from USAR GGSIPU interning at drdo"))

shell = ShellTool()
results= shell.invoke("mkdir weball")

print(results)
'''

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden",
)

@tool 
def multiply(a: int , b:int )-> int:
    """Multiplies two integeres and returns their product"""
    return a*b

llm = model.bind_tools([multiply])
result= llm.invoke("what is multiplication")

print(result)
