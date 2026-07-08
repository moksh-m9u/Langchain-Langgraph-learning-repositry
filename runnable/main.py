from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
 
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden"
)

prompt1 = PromptTemplate(
    template= "write a really funny and dark and smart joke on {topic} make sure to make it of only 2 lines",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "argue with him by roasting the joke {text} in just two lines",
    input_variables=['text']
)

parser= StrOutputParser()


joke_gen_chain = RunnableSequence(prompt1 , model , parser)

parallel = RunnableParallel({
    "joke": RunnablePassthrough(),
    "reply": RunnableSequence(prompt2 , model , parser)
})

chain = RunnableSequence(joke_gen_chain,parallel)

print(chain.invoke({"topic":"south delhi fat girl with red hairs"}))