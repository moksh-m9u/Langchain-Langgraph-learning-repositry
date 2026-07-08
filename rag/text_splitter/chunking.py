from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def splitting(documents: list[Document]) -> list[Document]:
    """
    Split LangChain documents into smaller overlapping chunks.

    Args:
        documents (list[Document]): Input documents.

    Returns:
        list[Document]: Chunked documents.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)