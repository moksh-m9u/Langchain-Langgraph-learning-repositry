# LangChain & LangGraph Learning Repository

**A hands-on exploration of LangChain and LangGraph, from foundational concepts to agentic workflows, RAG pipelines, and graph-based state machines.**

<div align="center">

[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langgraph&logoColor=white)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Ragas](https://img.shields.io/badge/Ragas-Evaluation-6E56CF?style=for-the-badge)](https://docs.ragas.io/)

</div>

<br/>

This is a personal, hands-on exploration of **LangChain** and **LangGraph** from foundational concepts to advanced agentic workflows, RAG pipelines, and graph-based state machines. A learner's playground, not a polished project or course. Expect rough edges, experiments, and work-in-progress code as I continue building.

I plan to keep learning new patterns and eventually polish this into a structured learning resource for beginners.

<br/>

## Table of Contents

- [Status](#status)
- [Topics Covered](#topics-covered)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)

<br/>

## Status

This repository is a **living document** of my learning journey. It gets updated as I learn new patterns and techniques. The code works but reflects the iterative nature of self-study — some files are exploratory, some are incomplete, and all of them taught me something.

Eventually I plan to polish this into a structured learning resource for beginners.

<br/>

## Topics Covered

### LangChain Fundamentals

| Topic | What it demonstrates | Location |
|-------|---------------------|----------|
| **Chains** | Parallel execution — two LLM calls in parallel (notes + quiz), then merging | `langchain/chains/` |
| **Output Parsers** | Structured output via Pydantic, JSON parsing, sequential Q&A, async streaming | `langchain/output_parsers/` |
| **Runnables** | RunnableSequence, RunnableParallel, RunnableLambda — composing chains with custom functions | `langchain/runnable/` |
| **Document Loaders** | Loading PDFs (PyPDFLoader) and web pages (WebBaseLoader) into Document objects | `langchain/document_loaders/` |
| **Text Splitters** | RecursiveCharacterTextSplitter — chunking strategies for long documents | `langchain/text_splitter/` |
| **Tools** | Custom @tool decorator + bind_tools() | `langchain/tools/` |
| **Practice Exercises** | Extra sequential, parallel, and structured output demos | `langchain/practise/` |

### LangGraph State Machines

| Pattern | What it demonstrates | Location |
|---------|---------------------|----------|
| **Sequential** | Prompt chaining, sequential LLM calls in a graph | `langgraph/fundamentals/Sequentials/` |
| **Parallel** | Essay evaluation, cricket performance — branching and joining nodes | `langgraph/fundamentals/Parallels/` |
| **Conditional** | Code review routing, customer support branching, quadratic equation solver | `langgraph/fundamentals/conditionals/` |
| **Iterative** | X/Twitter post agent — loops within a graph | `langgraph/fundamentals/Iteratives/` |
| **Chatbot** | Basic chatbot with and without persistence (memory) | `langgraph/chatbot/` |
| **Persistence** | Time travel, state inspection with InMemorySaver | `langgraph/persistence/` |

### Chatbot Versions (Persistance , Traceability , Threading)

| Version | Key Feature | Location |
|---------|-------------|----------|
| **v1** | Basic Streamlit UI with single thread | `langgraph/chatbot-versioning/chatbot/` |
| **v2** | Multi-thread management, conversation switching | `langgraph/chatbot-versioning/chatbotv2/` |
| **v3** | SQLite persistence via SqliteSaver | `langgraph/chatbot-versioning/chatbotv3/` |
| **v4** | LangSmith integration (planned) | `langgraph/chatbot-versioning/chatbotv4/` |

### Agents

- Research agents with tool-calling: web search + page scraping + source citation
- Streamlit interactive UI wrapping the research agent
- Demonstrates create_agent(), tool binding, and streaming
- Located in `langchain/agents/`

### RAG Pipeline (Retrieval-Augmented Generation)

| Stage | What it does | Location |
|-------|-------------|----------|
| **Document Loading** | Loads YouTube transcripts via YoutubeLoader | `langchain/rag/document_loading/` |
| **Text Splitting** | Chunks documents (1000 chars, 200 overlap) | `langchain/rag/text_splitter/` |
| **Embedding** | Dense vectors via HuggingFace (all-MiniLM-L6-v2 / bge-m3) | `langchain/rag/embedding/` |
| **Document Ingestion** | Load → chunk → embed → store in Chroma | `langchain/rag/document_ingestion/` |
| **Retrieval & Generation** | Top-k retrieval + Groq LLM for answers | `langchain/rag/main.py` |
| **Evaluation** | RAGAS scoring (faithfulness, answer correctness) | `langchain/rag/scoring.py` |

<br/>

## Project Structure

```
langgraph_practise/
├── .env.example                         # Template for required env vars
├── requirements.txt                     # Python dependencies
├── data/                                # Data files (PDFs, transcripts, blogs)
│   ├── datasheet.pdf
│   ├── dl-curriculum.pdf
│   ├── blog.txt
│   └── transcript.json
├── eval-scores/                         # RAGAS evaluation outputs
│   ├── score.csv
│   ├── score2.csv
│   └── score3.csv
│
├── langchain/                           # LangChain fundamentals & patterns
│   ├── main.py                          # Entry point — structured output demo
│   ├── agents/                          # Tool-calling agents + Streamlit UI
│   │   ├── agents.py
│   │   ├── intelligence.py
│   │   └── streamlit.py
│   ├── chains/                          # Chain patterns
│   │   ├── parallels.py
│   │   └── web_loader.py
│   ├── document_loaders/                # PDF & web document loading
│   │   ├── text_loaders.py
│   │   └── web_loader.py
│   ├── output_parsers/                  # Structured output, JSON, streaming
│   │   ├── basic.py
│   │   ├── basic2.py
│   │   └── json_parser.py
│   ├── practise/                        # Extra practice exercises
│   │   ├── main.py
│   │   ├── parallel.py
│   │   ├── sequential.py
│   │   └── structured_output_parser.py
│   ├── rag/                             # Full RAG pipeline
│   │   ├── main.py
│   │   ├── scoring.py
│   │   ├── document_loading/
│   │   ├── document_ingestion/
│   │   ├── embedding/
│   │   └── text_splitter/
│   ├── runnable/                        # RunnableSequence, Parallel, Lambda
│   │   ├── lambda_runnable.py
│   │   └── main.py
│   ├── text_splitter/                   # Chunking demos
│   │   └── practise.py
│   └── tools/                           # Custom tool definitions
│       └── practise.py
│
└── langgraph/                           # LangGraph notebooks & chatbot apps
    ├── requirements.txt
    ├── chatbot/                         # Basic chatbot notebooks
    │   ├── basic_chatbot.ipynb
    │   └── basic_chatbot_persistence.ipynb
    ├── chatbot-versioning/              # v1-v4 chatbot versions with UIs
    │   ├── chatbot/                     # v1 — basic Streamlit UI
    │   ├── chatbotv2/                   # v2 — multi-thread management
    │   ├── chatbotv3/                   # v3 — SQLite persistence
    │   └── chatbotv4/                   # v4 — LangSmith (planned)
    ├── fundamentals/
    │   ├── Sequentials/                 # 3 notebooks
    │   ├── Parallels/                   # 2 notebooks
    │   ├── conditionals/                # 3 notebooks
    │   └── Iteratives/                  # 1 notebook
    └── persistence/
        └── chatbot_persistance.ipynb
```

<br/>

## Getting Started

### Prerequisites

- Python 3.10+
- API keys: Groq (primary), Gemini, HuggingFace, Unstructured

### Installation

```bash
git clone <repo-url>
cd langgraph_practise

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
```

Fill in your API keys:

```env
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
OPENROUTER_API_KEY=your_openrouter_key_here
UNSTRUCTURED_API_KEY=your_unstructured_key_here
```

### Running Things

All commands run from the project root.

```bash
# LangChain demos
python langchain/main.py                         # Structured output demo

# LangChain scripts with internal imports (use -m from root)
python -m langchain.rag.main                     # RAG pipeline
python -m langchain.rag.scoring                  # RAGAS evaluation

# Streamlit UI
streamlit run langchain/agents/streamlit.py      # Research agent UI

# Chatbot UIs (use -m from chatbot-versioning/)
cd langgraph/chatbot-versioning && streamlit run chatbot/streamlit_frontend.py    # v1
cd langgraph/chatbot-versioning && streamlit run chatbotv2/streamlit_frontend.py  # v2
cd langgraph/chatbot-versioning && streamlit run chatbotv3/streamlit_frontend.py  # v3

# Chatbot FastAPI backend
cd langgraph/chatbot-versioning && uvicorn chatbot.fastapi_backend:app --reload

# Jupyter notebooks
jupyter notebook langgraph/chatbot/basic_chatbot.ipynb
```

<br/>

## Roadmap

- [x] LangChain fundamentals (chains, parsers, runnables, loaders, splitters, tools)
- [x] LangGraph state machines (sequential, parallel, conditional, iterative, chatbot)
- [x] Tool-calling agents
- [x] Full RAG pipeline with RAGAS evaluation
- [ ] More advanced agent patterns (reflection, planning, multi-agent)
- [ ] Graph RAG
- [ ] Production patterns (tracing, monitoring, guardrails)
- [ ] Polish into beginner-friendly resource with explanations

<br/>

## Tech Stack

| | |
|---|---|
| **LLM Frameworks** | LangChain, LangGraph |
| **LLM Providers** | Groq, Gemini, OpenRouter, HuggingFace |
| **Orchestration** | LangGraph state machines |
| **RAG** | Loading → chunking → embedding → ingestion → retrieval → evaluation |
| **Vector Store** | Chroma |
| **Evaluation** | RAGAS |
| **UI** | Streamlit, FastAPI |
| **Parsing** | Pydantic, JSON output parsers |

<br/>

<div align="center">

Built by [Moksh](https://github.com/moksh-m9u)

</div>
