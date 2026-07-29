from langchain_huggingface import HuggingFaceEndpointEmbeddings
from typing import List
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3"
)
'''
vector = embeddings.embed_query(
    "Explain transformers in simple terms."
)
#print (vector)
print(type(vector))
'''
def create_embeddings(query : str)-> List[float]:
    '''This function takes in a string and returns dense vector represntation in form of a List of Floats of length 1024
        args: 
            query : string = example : Explain transformer in simple terms
        return:
            List[float] : [0.100,2.34000,4.600,....... upto 1024 vectors]'''
    vector = embeddings.embed_query(query)
    return vector
    

