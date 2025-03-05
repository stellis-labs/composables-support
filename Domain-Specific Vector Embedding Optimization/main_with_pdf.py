import os
import chromadb
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from PyPDF2 import PdfReader
from typing import List, Dict, Union

class DatasetSynthesisModule:
    def __init__(self, 
                 knowledge_sources: Dict[str, Union[List[str], str]], 
                 prompt_file: str,
                 llm_name: str = "groq",
                 embedding_model_name: str = "all-MiniLM-L12-v2",
                 api_key: str = None):
        """
        Initialize the DatasetSynthesisModule.
        
        Args:
            knowledge_sources: Dict with keys 'urls' and 'pdf_dir' containing lists of URLs and PDF directory path
            prompt_file: Path to file containing prompts
            llm_name: Name of the LLM to use
            embedding_model_name: Name of the embedding model
            api_key: API key for the LLM
        """
        self.knowledge_sources = knowledge_sources
        self.prompt_file = prompt_file
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key
        self.vector_store = None
        self.index = None
        self.llm = None
        self.documents = []

        self.setup()

    def setup(self):
        """Set up the necessary components including document loading and index creation."""
        # Load documents from both URLs and PDFs
        self.load_documents_from_urls()
        self.load_documents_from_pdfs()

        # Set up embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Set up LLM
        if self.llm_name == "groq":
            self.llm = Groq(model="deepseek-r1-distill-qwen-32b", api_key=self.api_key)
        else:
            raise ValueError("Invalid LLM name. Only 'groq' is supported for now.")
        
        # Configure settings
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

        # Set up vector store
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = chroma_client.get_or_create_collection("financial_data")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from all documents
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=storage_context
        )

    def load_documents_from_urls(self):
        """Load and process documents from URLs."""
        if 'urls' not in self.knowledge_sources:
            return
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
            
        for url in self.knowledge_sources['urls']:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                text = soup.get_text(separator=" ", strip=True)
                document = Document(
                    text=text,
                    metadata={"source": url, "type": "url"}
                )
                self.documents.append(document)
                print(f"Loaded document from URL: {url}")
            except Exception as e:
                print(f"Failed to load/process document from {url}: {e}")

    def load_documents_from_pdfs(self):
        """Load and process documents from PDF files."""
        if 'pdf_dir' not in self.knowledge_sources:
            return
            
        pdf_dir = self.knowledge_sources['pdf_dir']
        if not os.path.exists(pdf_dir):
            print(f"PDF directory {pdf_dir} does not exist")
            return

        try:
            # Process each PDF file individually
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(pdf_dir, filename)
                    try:
                        reader = PdfReader(file_path)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        
                        if text.strip():  # Only create document if text was extracted
                            document = Document(
                                text=text,
                                metadata={
                                    "file_name": filename,
                                    "source": file_path,
                                    "type": "pdf"
                                }
                            )
                            self.documents.append(document)
                            print(f"Loaded PDF document: {filename}")
                        else:
                            print(f"Warning: No text extracted from {filename}")
                            
                    except Exception as e:
                        print(f"Failed to process PDF file {filename}: {e}")
                        
        except Exception as e:
            print(f"Failed to access PDF directory {pdf_dir}: {e}")

    def generate_dataset(self, num_examples=3):
        """Generate dataset by querying the knowledge base with prompts."""
        with open(self.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        
        if not prompts:
            print("No prompts found in the file. Cannot generate dataset.")
            return []
        
        dataset = []
        query_engine = self.index.as_query_engine()

        for i in range(num_examples):
            prompt = prompts[i % len(prompts)]
            response = query_engine.query(prompt)
            dataset.append({
                "text1": prompt,
                "text2": str(response),
                "metadata": {
                    "prompt_index": i % len(prompts),
                    "example_index": i
                }
            })
        return dataset

    def save_dataset(self, dataset, output_file):
        """Save the generated dataset to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    if not os.path.exists("prompts.txt"):
        with open("prompts.txt", "w") as f:
            f.write("What are the risks of investing in the stock market?\n")
            f.write("What are the basics of cryptocurrency?\n")
            f.write("What are the cryptocurrencies that are worth investing in 2025?\n")
            f.write("What is the summary of the Stock of Amazon?\n")

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Define knowledge sources including both URLs and PDF directory
    knowledge_sources = {
        'urls': [
            "https://www.investopedia.com/terms/c/cryptocurrency.asp",
            "https://www.fool.com/investing/stock-market/market-sectors/financials/cryptocurrency-stocks/next-crypto-to-explode/",
            "https://www.investors.com/research/magnificent-seven-stocks-february-2025/"
        ],
        'pdf_dir': '/Users/miteshsingh/Downloads/finance_pdfs'  # Directory containing PDF files
    }

    synthesis_module = DatasetSynthesisModule(
        knowledge_sources=knowledge_sources,
        prompt_file="prompts.txt",
        llm_name="groq",
        embedding_model_name="all-MiniLLM-L12-v2",
        api_key=groq_api_key
    )

    dataset = synthesis_module.generate_dataset(num_examples=60)
    for item in dataset:
        if "text2" in item:
            item["text2"] = item["text2"].replace("<think>", "").replace("</think>", "").strip()
    
    synthesis_module.save_dataset(dataset, "output25.json")