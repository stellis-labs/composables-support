# file for synthetic dataset generation using LLM and knowledge base
import os
import chromadb
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
# from llama_index.core import SimpleDirectoryReader # for documents loading
from llama_index.vector_stores.chroma import ChromaVectorStore

class DatasetSynthesisModule:
    def __init__(self, knowledge_links, prompt_file, llm_name="groq", embedding_model_name="all-MiniLM-L12-v2", api_key=None):
        self.knowledge_links = knowledge_links
        self.prompt_file = prompt_file
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key
        self.vector_store = None ##chromadb vector store instance
        self.index = None ##llamaindex
        self.llm = None
        self.documents = [] ##document store instance

        self.setup()

    def setup(self):
        self.load_documents_from_urls()
        chroma_client = chromadb.PersistentClient(path="./chroma_db") #chromadb local storage
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # documents = SimpleDirectoryReader(self.knowledge_links).load_data() 
        if self.llm_name=="groq":
            self.llm = Groq(model="deepseek-r1-distill-qwen-32b", api_key=self.api_key)
        else:
            raise ValueError("Invalid LLM name. Only 'groq' is supported for now.")
        
        # Create Settings object
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

        # vector store 
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("financial_data")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # create index
        self.index = VectorStoreIndex.from_documents(
            self.documents, 
            storage_context=storage_context
        )

    def load_documents_from_urls(self):
        for url in self.knowledge_links:
            try:
                response = requests.get(url)
                response.raise_for_status() #error raise for 4** and 5** status codes
                # parse data
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text(separator=" ", strip=True)
                # llama index document
                document = Document(    
                    text=text,
                    metadata={"source": url}
                )
                self.documents.append(document)
                print(f"Loaded document from {url}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to load document from {url}:{e}")
            except Exception as e:
                print(f"Failed to process document from {url}:{e}")

    def generate_dataset(self, num_examples=3):
        with open(self.prompt_file, "r") as f:
            prompts = [ line.strip() for line in f.readlines() ]
        print("prompts:***** ", prompts)
        dataset = []

        if not prompts:
            print("No prompts found in the file. Cannot generate dataset.")
            return dataset # returning empty dataset
        
        query_engine = self.index.as_query_engine()

        for i in range(num_examples):
            prompt = prompts[i % len(prompts)] #cycle through prompts
            response = query_engine.query(prompt) #query knowledge base
            text1 = prompt
            text2 = str(response)
            dataset.append({"text1": text1, "text2": text2})
        return dataset
    
    def save_dataset(self, dataset, output_file):
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset saved to {output_file}")
        

if __name__ == "__main__":
    if not os.path.exists("prompts.txt"):
        with open("prompts.txt", "w") as f:
            f.write("What are the risks of investing in the stock market?\n")
            f.write("What are the basics of cryptocurrency?\n")
            f.write("What are the cryptocurrencies that are worth investing in 2025?\n")

    load_dotenv()

    # groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")

    # knowlegde links
    knowledge_base_links = [
        # "https://www.investopedia.com/ask/answers/12/investing-in-the-stock-market.asp
        # ",
        "https://www.investopedia.com/terms/c/cryptocurrency.asp",
        "https://www.fool.com/investing/stock-market/market-sectors/financials/cryptocurrency-stocks/next-crypto-to-explode/"
    ]

    synthesis_module = DatasetSynthesisModule(
        knowledge_links=knowledge_base_links,
        prompt_file="prompts.txt",
        llm_name="groq",
        embedding_model_name="all-MiniLLM-L12-v2",
        api_key=groq_api_key
    )

    dataset = synthesis_module.generate_dataset(num_examples=10)
    for item in dataset:
        item["text2"] = item["text2"].replace("<think>", "").replace("</think>", "").strip() #think tags removed
    synthesis_module.save_dataset(dataset, "output.json")

