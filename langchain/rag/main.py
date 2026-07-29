from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.rag.document_loading.transcript import transcript
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.rag.text_splitter.chunking import splitting
import json

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="hidden",
)
parser = StrOutputParser()

docs = transcript("https://youtu.be/HTUh0OO6Kmo")
with open("data/transcript.json", "w", encoding="utf-8") as f:
    json.dump(
        [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs],
        f,
        indent=2,
        ensure_ascii=False
    )
chunks = splitting(docs)

vector_store = Chroma.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
'''
print(retriever)

print(retriever.invoke("Vibe coders"))
'''


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | model | parser

answer = main_chain.invoke('summarize the video and who is the creator')
print(answer)