from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot.langgraph_workflow import chatbot
from langchain_core.messages import HumanMessage
from datetime import datetime, timezone
import time

app = FastAPI(
    title="ChatBot API",
    description="A production-ready REST API for LLM Inference.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"


class ChatResponse(BaseModel):
    reply: str
    metadata: dict


@app.get('/')
def home():
    return {'status': 'OK'}



@app.get('/health')
def health_check():
    return {'status': 'OK'}


@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    query = [HumanMessage(content=req.message)]

    start = time.time()
    response = chatbot.invoke({"messages": query}, config=config)
    latency = round(time.time() - start, 3)

    ai_message = response['messages'][-1].content
    total_messages = len(response['messages'])

    return ChatResponse(
        reply=ai_message,
        metadata={
            "thread_id": req.thread_id,
            "total_messages": total_messages,
            "model": "llama-3.3-70b-versatile",
            "latency_seconds": latency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )