from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="qwen/qwen3-32b", temperature=0, max_tokens=None)


class Facts(BaseModel):
    facts: list[str] = Field(description="List of interesting facts about the topic")


structured_model = model.with_structured_output(Facts)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

chain = prompt | structured_model
result = chain.invoke({"topic": "LangChain"})
print(result)
