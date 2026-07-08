from datasets import Dataset
import os
from ragas import evaluate
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from ragas.metrics import faithfulness, answer_correctness
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from ragas import evaluate, RunConfig
load_dotenv()
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="hidden",
    max_tokens=None
)
ollama_llm = ChatOllama(
    model="qwen2:7b",
    temperature=0,
    request_timeout=6000000
)

evaluator_llm = LangchainLLMWrapper(llm)

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

data_samples = {"question": ["What effect do AI tools have on the learning gap between struggling and non-struggling programming students?","How did the struggling student in the study misuse the AI tool during her assignment?","What is the 'illusion of competence' described in the transcript?","How did the experienced student use AI autocomplete differently from the struggling student?","What was the methodology of the study that observed these students?","What is the recommended rule of thumb for using AI tools in programming?","How does coding without AI affect a programmer's confidence and sense of ownership?","When is it acceptable to use AI tools even if you are still learning to code?"],

"answer": [
    "AI tools can widen the learning gap — students with prior experience use them efficiently and move faster, while students without foundational knowledge tend to become more dependent on AI, producing working code without true understanding.",
    "The struggling student skipped the AI's explanations in favour of copying generated code directly. When the code didn't match her problem or contained unfamiliar syntax, she passed it back and forth between the IDE and the LLM until tests passed — without understanding how the solution worked.",
    "The illusion of competence is when a student believes they understand the material because AI helped them produce a working solution, but they cannot actually explain the logic behind it. They don't know what they don't know.",
    "The experienced student mostly ignored autocomplete suggestions, only accepting small, deliberate ones such as closing brackets or simple conditionals. She followed her own plan rather than letting the AI drive the solution.",
    "The study observed 21 students in the same introductory programming class completing the same assignment. Researchers used eye tracking and asked students to verbalise their thought process out loud (think-aloud protocol).",
    "The rule of thumb is: anything you would ask an AI to write, you should be able to write yourself. Doing things slowly without AI is better for learning and comprehension, similar to how handwritten notes are scientifically proven better for memory.",
    "Writing code without AI builds self-confidence and resilience to imposter syndrome. It also creates a sense of ownership — when you generate code yourself, you feel proud when it works and more motivated to debug when it doesn't. AI-generated code can feel 'meh', especially for beginners who need that positive feedback loop.",
    "It is acceptable to use AI when your goal is delivery rather than learning — for example, when a personal project is taking far longer than expected. In that case, the tradeoffs should be kept in mind: using AI to shortcut a problem means deliberately deprioritising deeper understanding of that component."
],

"contexts": [
    # Q1 — widening gap
    [
        "Researchers have noticed something weird. Some learners who use AI tools experience no change in their learning speeds. Maybe they're even a little faster in implementing their solutions. Others throw themselves at these tools and get worse.",
        "The main findings of the paper suggest that generative AI can create a widening gap between people who find programming easy and people who don't. Other studies have backed this up with some observing a higher failure rate in their courses after the introduction of these tools.",
        "This means that foundational concepts and prior experience are not just nice to have, but necessary for successfully using AI tools in programming. Otherwise, you're going to find yourself on the wrong side of that gap."
    ],

    # Q2 — struggling student misuse
    [
        "The student knows it's better to use AI as a tutor rather than an answer machine, so she requests step-by-step guidance, not the full answer. However, after reading the first point of advice, she scrolls past the explanation in search of a quicker solution and then copies the generated code.",
        "This code does not fully match the problem and contains some unfamiliar syntax leading to further confusion. After repeatedly passing her code back and forth between the IDE and the LLM, she eventually produces a version that passes the test cases, though without much understanding of how it works.",
        "Later, a researcher asks how useful she found the AI. The student answers, 'Very useful. And it's like having a personal tutor, and I really need that extra support in more difficult classes like this one.'"
    ],

    # Q3 — illusion of competence
    [
        "Here, we see the problem. The student lacks an awareness of the way that she actually used the AI. She has a skewed perception of her grasp of the material. An illusion of competence clouds her understandings. In short, she doesn't know what she doesn't know.",
        "They might be able to generate working code and they might even think they genuinely understand the problem, but ask them a couple questions like, 'Can you explain why this approach would fail for larger inputs?' or 'Why did you use a binary search tree over a linked list?' and their answers might get a little confused."
    ],

    # Q4 — experienced student
    [
        "When completing the same assignment, this student carefully reads the problem description, pauses to think, and begins structuring her solution. She knows she'll need certain variables to store the result and count, and she adds some comments to outline her strategy.",
        "As she types, the AI autocomplete frequent suggestions, which she occasionally glances at but mostly ignores, preferring to follow her own plan. When she does accept suggestions, they are small and deliberate, filling in predictable elements like closing brackets or simple conditionals.",
        "While she produces a few errors and syntax mistakes, she methodically corrects them and successfully completes the assignment, passing all test cases with flying colors. She acknowledges the AI helped her move faster, but credits her own thinking for getting to the right solution."
    ],

    # Q5 — study methodology
    [
        "Our subjects were two of 21 students in the same introductory programming class who were observed by researchers with eye tracking and asked to verbalize their thought process out loud while completing the same assignment.",
        "In this course, the teacher allows for the use of large language models and AI code completion software to help students with their work. They figure it's not going away, so we might as well learn to work with it."
    ],

    # Q6 — rule of thumb
    [
        "No matter whether you consider yourself a struggling or non-struggling student, you should assume that it is good for you to be able to code without AI tools. A simple rule of thumb is anything you'd ask an AI to write, you should be able to write yourself.",
        "Doing things slowly is better for learning, like how writing your notes by hand is scientifically proven to be better for memory and comprehension.",
        "We know that making sure you're not on the wrong side of that gap can probably mean an emphasis on writing more code by hand, at least in the beginning."
    ],

    # Q7 — confidence and ownership
    [
        "Additionally, coding with nothing but your own intelligence can improve your self-confidence, making you more resilient to imposter syndrome. It also changes the actual experience of coding.",
        "When you generate code with AI, you feel less ownership over it. This means it can be more frustrating when it doesn't work and less satisfying when it does.",
        "When you let something else do your thinking for you, you don't feel the same rush of satisfaction when the solution works. You don't feel smarter or proud of yourself, you just copy the solution and it worked, just like it was supposed to. It feels meh.",
        "If you're a beginner, this can eat away at the few experientially rewarding parts of coding in the time when you need that positive feedback the most. Not everyone can sit through hours of meh."
    ],

    # Q8 — when AI use is acceptable
    [
        "What about using AI when your goal is not learning, but doing? I get it. It's a big field and you might not always have the time or energy to invest in learning every little framework from scratch.",
        "If a personal project you thought you could finish in a few weeks is taking months to complete, I understand the temptation to vibe code the back end and come back to it later. In this case, I'm not completely against using it.",
        "When you can't help it, just keep these tradeoffs you're making in mind. Foster an awareness of when your mind is using the LLM to take a shortcut that will make you a worse programmer in the long run.",
        "My general advice is to not use LLMs when you can help it until you feel confident in your own ability."
    ]
],

"ground_truth": [
    "Generative AI widens the learning gap: students with prior experience leverage it to move faster, while students without foundational knowledge become dependent on it, producing working code without genuine understanding. Some courses have observed higher failure rates after introducing these tools.",
    "The struggling student skipped the AI's explanations and directly copied generated code. When the code mismatched her problem or had unfamiliar syntax, she repeatedly passed it between her IDE and the LLM until test cases passed — ending with a working solution she didn't understand.",
    "The illusion of competence occurs when a student feels they understand the material because AI produced a working solution, but they cannot explain the reasoning behind it. When asked deeper questions about their code choices, their understanding falls apart.",
    "The experienced student largely ignored AI autocomplete suggestions, only accepting small, targeted ones like closing brackets. She pre-planned her solution with comments, debugged her own errors methodically, and credited her own thinking — not the AI — for the final solution.",
    "The study involved 21 students in an introductory programming class, observed while completing the same assignment. Researchers used eye tracking and a think-aloud protocol (students verbalised their thought process out loud) to understand how each student interacted with AI tools.",
    "The recommended rule of thumb is: if you would ask an AI to write something, you should be able to write it yourself. Coding by hand, even slowly, is better for long-term learning and comprehension — analogous to how handwriting notes aids memory better than typing.",
    "Coding without AI builds self-confidence and helps combat imposter syndrome. It also creates a stronger sense of ownership over the code — when things break, the debugging is more motivated, and when things work, the satisfaction is genuine. AI-generated code removes this emotional feedback loop, which is particularly harmful for beginners who need positive reinforcement to stay engaged.",
    "Using AI is acceptable when the goal is delivery rather than learning — for example, when a project is significantly overrunning its timeline. However, this tradeoff must be consciously acknowledged: the shortcut means deliberately choosing not to deeply understand that part of the codebase for now."
]

}

dataset = Dataset.from_dict(data_samples)
N = 2

small_dataset = Dataset.from_dict({
    key: value[:N]
    for key, value in data_samples.items()
})
score = evaluate(small_dataset,metrics=[faithfulness,answer_correctness],llm=evaluator_llm,embeddings=ragas_embeddings)
df = score.to_pandas()
df.to_csv('score.csv',index=False)