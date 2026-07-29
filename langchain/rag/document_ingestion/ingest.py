from langchain.rag.document_loading.transcript import transcript
from langchain.rag.text_splitter.chunking import splitting
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


def ingest_youtube(url: str, collection_name: str = "default"):
    docs = transcript(url)
    chunks = splitting(docs)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        chunks, embeddings, collection_name=collection_name
    )
    return vector_store
