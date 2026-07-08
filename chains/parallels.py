from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1= ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed"
)
model2 = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed"
)

prompt1 = PromptTemplate(
    template="You are a expert teacher to explain any topic in {domain} from first principle and prepare them conceptually on {topic}",
    input_variables=['topic','domain']
)
prompt2 = PromptTemplate(
    template="You are a expert teacher to asses students for any topic in {domain} from first principle and prepare a quiz of  of 5 questions from  {topic}",
    input_variables=['topic','domain']
)
prompt3= PromptTemplate(
    template="Merge the given notes and question into well formatted and structered way \n notes : {notes} , Quiz : {quiz}"
)

parser= StrOutputParser()

parallel = RunnableParallel({
    "notes":prompt1 | model1 | parser,
    "quiz":prompt2| model2| parser
})

bottom = prompt3 | model1 | parser

chain = parallel | bottom

domain = "MLOps Dev"
text = """
        Well, most of us are aware about Git and how it can be used as a version control system for our entire codebase.

        however, its just worth to ensure that we are on the same page
        every time we made a commit git stores that particular version of code to which we can rollback if required.

        With each commit git generates a sha-id which can be essentially used to roll back by running the command

        git checkout <sha-id-of-your-commit>

        Pretty powerful right?

        Well, For most of the part yes!
        But if we talk about Machine Learning Systems , where we have data files whose size is generally larger than what essentially git can track . it becomes impossible for us to track those files using git

            Lets have a look at the root directory of very basic ML system 
        credit-card-fraud-detection-system/
        ├── data/
        │   ├── raw/fraud_dataset.csv          # 150MB
        │   └── processed/train_scaled.npy     # 200MB
        ├── models/
        │   └── xgboost_v3_recall_tuned.pkl    # 400MB
        ├── notebooks/
        │   └── eda.ipynb
        └── src/
            └── train.py

        Now try running git add data/ on that 150MB CSV.

            GitHub will reject files over 100MB.

            Even if it didn’t do you want to store binary blobs in your Git history?

            Every commit would balloon in size.

            Cloning the repo would take minutes.

            Diffs would be meaningless (you can’t diff a .pkl file)

        The core problem:

            Git is built for code

        ML projects are code + data + models + metrics , and these are interdependent on each other , a change or update in data would result in different metrics over time.

        also, while training models our interest lies in experimenting with different set of parameters, models , and data splits to get the best possible model for a given problem.

        Git is not built for tracking data , models and providing a modular way to track our experiments 
        """

chain.get_graph().print_ascii()
result = chain.invoke({"domain":domain, "topic":text})
print (result)