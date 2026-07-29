from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage , HumanMessage,SystemMessage,BaseMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict , List , Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0
)

'''Adding reducer function add_messages from langgraph.graph.message its a reducer function
to add consequent messages in our State Variable ChatBot['messages']'''
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return{
        'messages':[response]
    }

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)