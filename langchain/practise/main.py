from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

# -------------------------
# Models
# -------------------------

model1 = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed"
)

model2 = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed"
)

# -------------------------
# Prompts
# -------------------------

prompt1 = PromptTemplate(
    template="""
You are an expert teacher.

Explain the topic from first principles.

Domain: {domain}

Topic:
{topic}
""",
    input_variables=["domain", "topic"]
)

prompt2 = PromptTemplate(
    template="""
You are an expert examiner.

Create 5 quiz questions.

Domain: {domain}

Topic:
{topic}
""",
    input_variables=["domain", "topic"]
)

# -------------------------
# Parallel Branches
# -------------------------

parallel = RunnableParallel(
    {
        "notes": prompt1 | model1,
        "quiz": prompt2 | model2,
    }
)

# -------------------------
# Invoke
# -------------------------

result = parallel.invoke(
    {
        "domain": "MLOps Dev",
        "topic": "Git struggles with large datasets and model files."
    }
)

# =====================================================
# NOTES RESPONSE
# =====================================================

notes_msg = result["notes"]

print("\n" + "=" * 50)
print("NOTES CONTENT")
print("=" * 50)

print(notes_msg.content)

print("\n" + "=" * 50)
print("NOTES additional_kwargs")
print("=" * 50)

print(notes_msg.additional_kwargs)

print("\n" + "=" * 50)
print("NOTES response_metadata")
print("=" * 50)

print(notes_msg.response_metadata)

# =====================================================
# QUIZ RESPONSE
# =====================================================

quiz_msg = result["quiz"]

print("\n" + "=" * 50)
print("QUIZ CONTENT")
print("=" * 50)

print(quiz_msg.content)

print("\n" + "=" * 50)
print("QUIZ additional_kwargs")
print("=" * 50)

print(quiz_msg.additional_kwargs)

print("\n" + "=" * 50)
print("QUIZ response_metadata")
print("=" * 50)

print(quiz_msg.response_metadata)