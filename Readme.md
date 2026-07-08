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

This repository is a **living document** of my learning journey. It gets updated as I learn new patterns and techniques. The code works but reflects the iterative nature of self-study  some files are exploratory, some are incomplete, and all of them taught me something.

Eventually I plan to polish this into a structured learning resource for beginners.

<br/>

## Topics Covered

### LangChain Fundamentals

| Topic | What it demonstrates | Files |
|-------|---------------------|-------|
| **Chains** | Parallel execution — two LLM calls in parallel (notes + quiz), then merging | `chains/` |
| **Output Parsers** | Structured output via Pydantic, JSON parsing, sequential Q&A, async streaming | `output_parsers/` |
| **Runnables** | RunnableSequence, RunnableParallel, RunnableLambda — composing chains with custom functions | `runnable/` |
| **Document Loaders** | Loading PDFs (PyPDFLoader) and web pages (WebBaseLoader) into Document objects | `document_loaders/` |
| **Text Splitters** | RecursiveCharacterTextSplitter — chunking strategies for long documents | `text_splitter/` |
| **Tools** | Custom @tool decorator + bind_tools() | `tools/` |

### LangGraph State Machines

| Pattern | What it demonstrates | Files |
|---------|---------------------|-------|
| **Sequential** | Prompt chaining, sequential LLM calls in a graph | `langgraph/fundamentals/Sequentials/` |
| **Parallel** | Essay evaluation, cricket performance — branching and joining nodes | `langgraph/fundamentals/Parallels/` |
| **Conditional** | Code review routing, customer support branching, quadratic equation solver | `langgraph/fundamentals/conditionals/` |
| **Iterative** | X/Twitter post agent — loops within a graph | `langgraph/fundamentals/Iteratives/` |
| **Chatbot** | Basic chatbot with and without persistence (memory) | `langgraph/chatbot/` |

### Agents

- Research agents with tool-calling: web search + page scraping + source citation
- Streamlit interactive UI wrapping the research agent
- Demonstrates create_agent(), tool binding, and streaming

### RAG Pipeline (Retrieval-Augmented Generation)

| Stage | What it does | Files |
|-------|-------------|-------|
| **Document Loading** | Loads YouTube transcripts via YoutubeLoader | `rag/document_loading/` |
| **Text Splitting** | Chunks documents (1000 chars, 200 overlap) | `rag/text_splitter/` |
| **Embedding** | Dense vectors via HuggingFace (all-MiniLM-L6-v2 / bge-m3) | `rag/embedding/` |
| **Document Ingestion** | Load → chunk → embed → store in Chroma | `rag/document_ingestion/` |
| **Retrieval & Generation** | Top-k retrieval + Groq LLM for answers | `rag/main.py` |
| **Evaluation** | RAGAS scoring (faithfulness, answer correctness) | `rag/scoring.py` |

<br/>

## Project Structure

```
langgraph_practise/
├── .env.example            # Template for required env vars
├── agents/                 # Agent implementations
│   ├── agents.py           #   Research agent (web search + scrape)
│   ├── intelligence.py     #   Simple research agent
│   └── streamlit.py        #   Streamlit chat UI
├── chains/                 # Chain patterns
│   ├── parallels.py        #   Parallel notes + quiz generation
│   └── web_loader.py       #   Web document loading demo
├── document_loaders/       # PDF & web document loading
├── langgraph/              # LangGraph notebooks
│   ├── chatbot/            #   Basic chatbot notebooks
│   └── fundamentals/
│       ├── Sequentials/    #   3 notebooks
│       ├── Parallels/      #   2 notebooks
│       ├── conditionals/   #   3 notebooks
│       └── Iteratives/     #   1 notebook
├── notebooks/              # Miscellaneous notebooks
├── output_parsers/         # Structured output, JSON, streaming
├── practise/               # Extra practice exercises
├── rag/                    # Full RAG pipeline
│   ├── document_loading/
│   ├── text_splitter/
│   ├── embedding/
│   ├── document_ingestion/
│   ├── main.py             # Pipeline orchestrator
│   └── scoring.py          # RAGAS evaluation
├── requirements.txt        # Python dependencies
├── runnable/               # RunnableSequence, Parallel, Lambda
├── text_splitter/          # Chunking demos
├── tools/                  # Custom tool definitions
└── main.py                 # Entry point — structured output demo
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

```bash
python main.py                    # Structured output demo
python rag/main.py                # RAG pipeline
python rag/scoring.py             # RAGAS evaluation
streamlit run agents/streamlit.py # Agent UI
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
| **UI** | Streamlit |
| **Parsing** | Pydantic, JSON output parsers |

<br/>

<div align="center">

Built by [Moksh](https://github.com/moksh-m9u)

</div>
