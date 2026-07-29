from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
print(f"Loaded {len(docs)} document(s)")
print(docs[0].page_content[:500])
