# dataset_synthesis.py
import os
import chromadb
import logging
import argparse
import pandas as pd
import random
import json
import time
import re

from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import the separate modules
from url_knowledge_base import URLKnowledgeBase
from pdf_knowledge_base import PDFKnowledgeBase

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_synthesis.log", mode='w',encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetSynthesisModule:
    """Main module integrating URL and PDF knowledge bases into vector storage."""

    def __init__(self, use_url, use_pdf, knowledge_links, pdf_folder, llm_name="groq", embedding_model_name="all-MiniLM-L6-v2", api_key=None):
        self.use_url = use_url
        self.use_pdf = use_pdf
        self.llm_name = llm_name
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key
        self.documents = []
        self.synthetic_dataset = []

        # Load Paraphrasing Model
        self.paraphraser = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.paraphraser_tokenizer = T5Tokenizer.from_pretrained("t5-base")

        # Initialize knowledge sources
        if self.use_url:
            self.url_knowledge_base = URLKnowledgeBase(knowledge_links)
        if self.use_pdf:
            self.pdf_knowledge_base = PDFKnowledgeBase(pdf_folder)

        self.setup()

    def save_extracted_documents(self):
        """Save extracted document text to a file for verification."""
        with open('extracted_documents.txt', 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(f"Source: {doc.metadata['source']}\n")
                f.write(doc.text + "\n\n" + "="*80 + "\n\n")  # Separate documents
            logger.info("Extracted documents saved to extracted_documents.txt.")

    def setup(self):
        """Initialize components, process documents, and generate synthetic cryptocurrency dataset."""
        if self.use_url:
            self.url_knowledge_base.fetch_documents()
            self.documents.extend(self.url_knowledge_base.documents)

        if self.use_pdf:
            self.pdf_knowledge_base.extract_documents()
            self.documents.extend(self.pdf_knowledge_base.documents)

        if not self.documents:
            logger.warning("No documents available for indexing or synthetic data generation.")
            return 
        
        # filter out documents with non-meaningful text
        self.documents = [doc for doc in self.documents if len(doc.text.split()) > 20]
        
        # Save extracted documents to a file
        self.save_extracted_documents()
        
        # Initialize ChromaDB for vector storage
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Initialize LLM - llama-3.3-70b-versatile
        if self.llm_name == "groq":
            self.llm = Groq(model="llama-3.3-70b-specdec", api_key=self.api_key)
            # self.llm = Groq(model="llama-3.2-90b-vision-preview", api_key=self.api_key)
            # self.llm = Groq(model="llama-3.3-70b-versatile", api_key=self.api_key)
            # self.llm = Groq(model="qwen-2.5-32b", api_key=self.api_key) # risk in defi #pdfs
        else:
            raise ValueError("Invalid LLM name. Only 'groq' is supported for now.")

        # Set LlamaIndex settings
        Settings.llm = self.llm
        # Settings.embed_model = SentenceTransformer(self.embedding_model_name)
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)
        # Settings.chunk_size = 200  # Daily limit is kept in mind
        # Settings.chunk_size = 500 # defi knowledge base
        # Settings.chunk_size = 350 # ai in fin knowledge base
        Settings.chunk_size = 200 # ai in fin knowledge base

        # Create vector store
        chroma_collection = chroma_client.get_or_create_collection("financial_data")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from documents
        if self.documents:
            self.index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context)
            logger.info("Vector index created successfully.")
        
        # Generate synthetic financial dataset
        self.synthetic_dataset = self.generate_synthetic_dataset(num_samples=2500)

        # # Save dataset to CSV
        # if self.synthetic_dataset:
        #     df = pd.DataFrame({"synthetic_dataset": self.synthetic_dataset})
        #     df.to_csv("synthetic_dataset.csv", index=False)
        #     logger.info("Synthetic dataset saved to synthetic_dataset.csv.")

    def get_random_chunk(self, text, chunk_size=10000):
        """Select a random chunk of text from the document."""
        if len(text) <= chunk_size:
            return text  # Return the entire text if it's smaller than the chunk size
        start_index = random.randint(0, len(text) - chunk_size)
        return text[start_index:start_index + chunk_size]


    def generate_synthetic_dataset(self, num_samples=100):
        """Generate synthetic financial dataset using the LLM."""
        if not self.index:
            logger.warning("No vector index available.")
            return None

        # output_file = "synthetic_crypto_qa2.csv" #llama3.3 70b specdec
        # output_file = "synthetic_crypto_qa4.csv" #defi #deepseek-r1-distill-llama-70b
        # output_file = "synthetic_crypto_qa5.csv" #llama3.3 70b specdec
        # output_file = "synthetic_crypto_qa6.csv" #llama3.3 70b versatile # algorithmic trading
        output_file = "synthetic_crypto_qa7.csv" #deepseek-r1-distill-llama-70b versatile # risk in defi
        existing_questions = set()
        existing_question_embeddings = []  # Storing embeddings for similarity checking

        # For Semantic Similarity Checking
        embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize CSV file with headers
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("Question, Answer\n")

        # Load existing questions if the file already exists
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:  # Skip header row
                    try:
                        question = line.split('","')[0].strip('"')
                        existing_questions.add(question)
                        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
                        existing_question_embeddings.append(question_embedding)
                    except Exception as e:
                        logger.warning(f"Error processing line: {line}. Error: {e}")
                        continue

        synthetic_dataset = []
        requests_made = 0
        start_time = time.time()
        rate_limit_reached_flag = False

        for i in range(num_samples):
            if requests_made >= 30:  # Enforce 30 requests per minute
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                start_time = time.time()
                requests_made = 0

            # may not be the best way to select chunks of documents
            document_text = random.choice(self.documents).text[:5000]
            # Usage in the main loop
            # document_text = self.get_random_chunk(self.documents)
            logger.info(f"DOC_TEXT before: {document_text}")
            # Retry counter for highly similar questions
            similar_question_retries = 0

            for attempt in range(3):  # Retry logic for failed API calls
                try:
                    # print("📢 About to send prompt to LLM...")  # Debugging checkpoint
                    logger.info("📢 About to send prompt to LLM...")
                    response = self.llm.complete(
                        f"""Based on this finance knowledge, generate a **unique** question-answer pair that has not been seen before. 
                        Ensure the question is specific, concise, and covers a different aspect of the topic. 
                        Return **only** a JSON object in the following format:

                        {{"question": "...", "answer": "..."}}

                        Text:
                        {document_text}
                    """
                    )
                    print("📢 After LLM complete...")  # Debugging checkpoint
                    # logger.info(f"Final Prompt Sent to LLM:\n{document_text[:1000]}")  # Log first 1000 chars
                    response_text = response.text if hasattr(response, "text") else str(response)
                    logger.info(f"LLM RESPONSE for sample {i+1}:\n{response_text}")
                    qa_pair = json.loads(response_text.strip())  # Ensure valid JSON format

                    # Check if question is already in the dataset
                    if qa_pair["question"] in existing_questions:
                        logger.info(f"Duplicate question detected. Skipping: {qa_pair['question']}")
                        continue  # Skip duplicates

                    # Check for similar questions using embeddings (Semantic Similarity)
                    new_question_embedding = embedding_model.encode(qa_pair["question"], convert_to_tensor=True)

                    # if existing_question_embeddings:
                    #     similarity_scores = [util.pytorch_cos_sim(new_question_embedding, embedding)[0] for embedding in existing_question_embeddings]
                    #     if any(score > 0.10 for score in similarity_scores):  # Threshold for similarity earlier 85, 65
                    #         logger.info(f"Highly similar question detected. Skipping: {qa_pair['question']}")
                    #         similar_question_retries += 1
                    #         # doc_text = random.choice(self.documents).text[:10000] chnage
                    #         if similar_question_retries >= 1:
                    #             logger.info("Similar question detected twice. Selecting a new chunk of text.")
                    #             # document_text = random.choice(self.documents).text[:5000]
                    #             # random_document = random.choice(self.documents)  # Select a new document
                    #             document_text = random.choice(self.documents).text[:5000]
                    #         # if similar_question_retries >= 5:
                    #         #     logger.info("Not wasting APIs. Let's break. Check code and rerun.")
                    #         #     break
                    #         continue  # Skip semantically similar questions

                    # Save to file immediately
                    existing_questions.add(qa_pair["question"])
                    existing_question_embeddings.append(new_question_embedding)
                    
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(f'"{qa_pair["question"]}","{qa_pair["answer"]}"\n')

                    synthetic_dataset.append(qa_pair)
                    requests_made += 1
                    break  # Success, move to next sample

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON format for entry {i}. Skipping.")
                    logger.info("Invalid JSON format", response_text)
                    # Additionally will try to fix malformed JSON
                    break  # Skip if bad JSON

                except Exception as e:
                    error_message = str(e)
                    print("$$$ printing error message ", str(e).upper())
                    # rate_limit_reached_flag = False
                    if "rate limit reached" in error_message.lower() and ("requests per day" in error_message.lower() or "tokens per day" in error_message.lower()):
                        # Daily token limit reached
                        logger.info(f"ERRROOOORRR: Daily token limit reached. Stopping execution. Error: {error_message}")
                        rate_limit_reached_flag = True
                        break
                    else:
                        logger.warning(f"Request failed (Attempt {attempt+1}/3). Retrying in {2**attempt} seconds... Error: {e}")
                        time.sleep(2**attempt)  # Exponential backoff (2s, 4s, 8s)

            if i % 100 == 0:
                logger.info(f"Generated {i}/{num_samples} synthetic dataset entries.")

            if rate_limit_reached_flag:
                break # breaking as TPD/RPD reached

        return synthetic_dataset
    

if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Parse user input to choose sources
    parser = argparse.ArgumentParser(description="Choose knowledge sources.")
    parser.add_argument("--use-url", action="store_true", help="Use URL-based knowledge sources.")
    parser.add_argument("--use-pdf", action="store_true", help="Use PDF-based knowledge sources.")
    args = parser.parse_args()

    knowledge_base_links = [

        # Predictive Analytics in Finance
        "https://www.ibm.com/think/topics/predictive-analytics?utm_content=SRCWW&p1=Search&p4=43700075153304567&p5=p&p9=58700008227853819&gad_source=1&gclid=Cj0KCQiA8q--BhDiARIsAP9tKI1lITZmGEGWv5aJcAtVcC_UH0JHgUmNUiHjG0y1VNaPuO3m_iaY22AaAjkREALw_wcB&gclsrc=aw.ds",
        "https://www.highradius.com/resources/Blog/predictive-analytics-in-finance-guide/#:~:text=What%20is%20Predictive%20Analytics%20in,about%20future%20events%20or%20behaviors.",
        "https://www.bluent.com/blog/predictive-analytics-in-finance/",
        "https://www.velvetech.com/blog/predictive-analytics-in-finance/",
        "https://blog.workday.com/en-us/how-ai-is-shaping-predictive-analytics-in-finance.html",
        "https://www.linkedin.com/pulse/predictive-analytics-transforming-risk-management-ekkarit-7mn1c/",
        "https://morningconsult.com/2022/11/29/predictive-analytics-in-finance/",
        "https://www.datarails.com/predictive-analytics-in-finance/",
        "https://nowcfo.com/predictive-analytics-in-finance-the-hidden-gem-in-financial-planning/",
        "https://www.sciencedirect.com/science/article/pii/S2405918822000071",
        "https://www.fm-magazine.com/news/2023/may/4-ways-cfos-maximise-benefits-predictive-analytics/",
        "https://www.pwc.ch/en/insights/finance-transformation/predictive-analytics.html",
        "https://www.explo.co/blog/financial-predictive-analytics",
        "https://ramp.com/blog/predictive-analytics-in-finance/",
        "https://www.linkedin.com/pulse/artificial-intelligence-finance-predictive-smart-josyula-mba/",
        "https://sumatosoft.com/blog/predictive-analytics-in-finance-use-cases",
        "https://panintelligence.com/blog/predicitive-analytics-in-finance/",
        "https://yellow.systems/blog/predictive-analytics-in-finance",
        "https://svitla.com/blog/predictive-analytics-finance/",
        "https://www.jedox.com/en/blog/predictive-analytics/",
        "https://biztechmagazine.com/article/2022/02/what-predictive-analytics-and-how-can-it-help-financial-institutions-manage-risk",
        "https://online.hbs.edu/blog/post/predictive-analytics",
        "https://www.futureviewsystems.com/blog/predictive-vs-prescriptive-analytics-in-finance",
        "https://www.engineersmind.com/Services/AI-and-Data?gad_source=1&gclid=Cj0KCQiA8q--BhDiARIsAP9tKI2afm8bZAVIKsWLcVkrm1UC1MpX6u9I7WGBSaiCjTiocjYUes96EJwaArbmEALw_wcB",
        "https://innovyne.com/predictive-analytics-in-finance/",
        "https://www.dexmiq.com/blog-posts/predictive-analytics-in-finance-e083f",
        "https://blog.fabrichq.ai/predictive-analytics-in-finance-here-is-everything-you-should-know-70fba95b1cb5",
        "https://www.swapsupport.com/articles/predictive-analytics-finance",
        "https://www.qulix.com/about/blog/predictive-analytics-in-finance/",
        "https://www.softkraft.co/predictive-analytics-in-finance/",
        "https://saxon.ai/blogs/predictive-analytics-in-finance-use-cases-and-benefits/",
        "https://febi.ai/blog/how-to-use-predictive-analytics-for-financial-planning/",
        "https://www.finextra.com/blogposting/26980/al-driven-dashboards-how-predictive-al-analytics-is-shaping-the-future-financial-forecasting",
        "https://www.prioxis.com/blog/predictive-analytics-in-finance",
        "https://www.qlik.com/us/data-analytics/financial-analytics",
        "https://www.rapidinnovation.io/post/predictive-analytics-in-finance-anticipating-market-trends",
        "https://innovatureinc.com/the-advantage-of-predictive-analytics-in-finance/",
        'https://imarticus.org/blog/role-of-predictive-analytics-in-cash-flow-forecasting/'
        "https://www.zeni.ai/blog/financial-predictive-analytics",
        "https://neontri.com/blog/predictive-analytics-banking/",
        "https://www.pi.exchange/blog/predictive-analytics-in-banking",
        "https://binariks.com/blog/predictive-analytics-in-fintech/",
        "https://www.sciencedirect.com/science/article/pii/S2405918822000071?ssrnid=4053319&dgcid=SSRN_redirect_SD"
    ]
  
    pdf_folder = "pdfs_data"

    # Initialize the synthesis module based on the user's choice
    synthesis_module = DatasetSynthesisModule(
        use_url=args.use_url,
        use_pdf=args.use_pdf,
        knowledge_links=knowledge_base_links,
        pdf_folder=pdf_folder,
        llm_name="groq",
        embedding_model_name="all-MiniLM-L6-v2",
        api_key=groq_api_key
    )