
import os
from google.colab import userdata
import pandas as pd
from langchain_community.document_loaders import PubMedLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas.testset import TestsetGenerator
# from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

data_generation_model= ChatGroq(temperature=0.8, model_name="llama3-70b-8192" ,api_key="")
data_generation_model=LangchainLLMWrapper(data_generation_model)


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

!pip install --upgrade urllib3

loader = PubMedLoader("cancer", load_max_docs=5)

loader

documents = loader.load()
documents

generator = TestsetGenerator(llm= data_generation_model, embedding_model=embeddings)
dataset = generator.generate_with_langchain_docs(documents, testset_size=10)

dataset.to_pandas()
