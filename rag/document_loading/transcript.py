from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document

def transcript(url: str) -> list[Document]:
    """
    Load the transcript of a YouTube video.

    Args:
        url (str): YouTube video URL.

    Returns:
        list[Document]: Transcript represented as LangChain Documents.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(
            youtube_url=url,
            add_video_info=False,
        )

        docs = loader.load()

        return docs
    except Exception as e:
        raise RuntimeError(
            f"Failed to load transcript for YouTube URL: {url}"
        ) from e