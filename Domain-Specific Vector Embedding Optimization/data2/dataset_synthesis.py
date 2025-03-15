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
                    if "rate limit reached" in error_message.lower() and "requests per day" in error_message.lower():
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

    # 1-500
    # Define knowledge sources
    knowledge_base_links = [
        # cryptocurrency
        # "https://usa.kaspersky.com/resource-center/definitions/what-is-cryptocurrency",
        "https://en.wikipedia.org/wiki/Cryptocurrency",
        "https://coinmarketcap.com/",
        "https://www.oswego.edu/cts/basics-about-cryptocurrency",
        "https://www.bankrate.com/investing/top-performing-crypto/",
        "https://coinledger.io/learn/best-long-term-crypto",
        "https://www.forbes.com/sites/digital-assets/article/top-cryptocurrencies-to-watch-2025/",
        "https://investinghaven.com/crypto-forecasts/15-cryptocurrency-forecasts-2025/",
        "https://www.investopedia.com/best-crypto-exchanges-5071855",
        "https://www.pwc.com/us/en/industries/financial-services/fintech/bitcoin-blockchain-cryptocurrency.html",
        "https://www.rba.gov.au/education/resources/explainers/cryptocurrencies.html",
        "https://www.google.com/finance/markets/cryptocurrencies?hl=en"
        "https://www.edwardjones.com/us-en/investment-services/investment-products/how-do-stocks-work?utm_source=google&utm_medium=paidsearch&utm_campaign=20833985210&utm_agid=157081167795&utm_term=investing%20in%20stocks&creative=683386381392&device=c&mjp=exp&mbu=per&mau=na&mob=stc&mbt=gim&&&&&gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsII-iofxjgfP8SB3HPS6I4zj4AT9W9RfYwVFP8k7SNdlK39M5G9eBTxoChqMQAvD_BwE&gclsrc=aw.ds",
        "https://www.fidelity.com/learning-center/trading-investing/crypto/decentralized-finance-defined#:~:text=DeFi%20stands%20for%20decentralized%20finance,first%20cover%20traditional%2C%20centralized%20finance.",
        "https://smartvalor.com/ru/news/defi-basics",
        "https://www.blockpit.io/en-us/blog/what-is-defi-decentralized-finance",
        "https://www.argoblockchain.com/articles/defi-revolution-decentralized-finance",
        "https://disb.dc.gov/page/beware-decentralized-finance-defi",
        "https://chain.link/education/defi/",
        "https://www.coursera.org/articles/what-is-defi",
        "https://appinventiv.com/blog/decentralized-finance-defi-guide/",
        "https://academy.geniusyield.co/guides/defi",
        "https://ijaracdc.com/decentralized-finance/?utm_source=google&utm_campaign=22173774846&utm_content=&utm_term=&utm_medium=&gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIAo4izdu1B9Ai4wBSew2ehNLoLZIC8JjG0VfcNs8NrGl16xlarfStBoC9-oQAvD_BwE",
        "https://aws.amazon.com/web3/what-is-defi/",
        "https://n26.com/en-eu/blog/what-is-defi",
        "https://www.paystand.com/blog/decentralized-finance",

        # blockchain
        "https://denver-south.com/blog/what-is-a-blockchain-an-easy-to-digest-guide-for-the-uninitiated/?gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIG8i1BFKr3w4Ya5q0xfpCkoVr3mWBIVUCfqsVmi84Gzk_zw-jzPhuhoC4sIQAvD_BwE",
        "https://www.cnn.com/markets",
        "https://www.investors.com/market-trend/stock-market-today/dow-jones-futures-trump-blames-globalists-nvidia-tesla-palantir-broadcom-jobs-report/",
        "https://finance.yahoo.com/news/live/stock-market-today-nasdaq-enters-correction-sp-500-sinks-to-lowest-since-november-as-stocks-get-clobbered-on-trump-tariff-whiplash-210544344.html",
        "https://tradingeconomics.com/united-states/stock-market,"
        "https://www.google.com/finance/?hl=en",
        "https://www.nyse.com/index",
        "https://www.nasdaq.com/",
        "https://apnews.com/article/investors-stock-market-wall-street-volatility-a8bb85c802be802929bda213253c8178",
        "https://en.wikipedia.org/wiki/Stock_market",
        "https://www.google.com/finance/markets/indexes?hl=en",
        "https://www.wsj.com/market-data/stocks",
        "https://www.nytimes.com/section/markets-overview",
        "https://www.reuters.com/markets/us/",
        "https://www.wsj.com/livecoverage/stock-market-today-dow-sp500-nasdaq-live-03-06-2025",
        "https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/stock-market/",
        "https://www.bloomberg.com/markets",
        "https://www.schwab.com/learn/story/stock-market-update-open",
        "https://www.cnn.com/markets/after-hours",
        "https://www.foxbusiness.com/stocks",
        "https://www.morningstar.com/markets",
        "https://en.wikipedia.org/wiki/Stock_market_index",
        "https://www.nseindia.com/",
        "https://www.bseindia.com/",
        "https://www.dcreport.org/2024/06/18/stock-market-simplified-a-students-guide-to-understanding-investments/?gad_source=1&gclid=CjwKCAiArKW-BhAzEiwAZhWsIO6X38U5_9sQYE-6OINTeNNXlal0gyOWFHx31up28hg8pW8YwngNFxoCcGEQAvD_BwE",
        "https://www.td.com/ca/en/investing/direct-investing/articles/what-is-stock-market",
        "https://www.nyse.com/markets/nyse/trading-info",
        "https://edition.cnn.com/markets/premarkets",
        "https://business.fiu.edu/academics/graduate/insights/posts/artificial-intelligence-in-the-stock-market-how-did-it-happen.html",
        "https://www.bloomberg.com/news/articles/2025-03-05/stock-market-today-dow-s-p-live-updates",
        "https://jamapunji.pk/knowledge-center/how-trade-stock-market"

        # ai in finance
        "https://www.oecd.org/en/topics/sub-issues/digital-finance/artificial-intelligence-in-finance.html#:~:text=It%20is%20used%20in%20fraud,%2C%20trading%2C%20and%20risk%20analysis.",
        "https://cloud.google.com/discover/finance-ai",
        "https://cloud.google.com/discover/finance-ai#how-is-ai-used-in-finance",
        "https://cloud.google.com/discover/finance-ai#what-is-ml-in-finance",
        "https://cloud.google.com/discover/finance-ai#applications-how-ai-can-solve-real-challenges-in-financial-services",
        "https://cloud.google.com/discover/finance-ai#benefits-of-ai-in-finance",
        "https://cloud.google.com/discover/finance-ai#the-future-of-ai-in-financial-services",
        "https://cloud.google.com/discover/finance-ai#hear-from-our-customers",
        "https://cloud.google.com/discover/finance-ai#related-products-and-services",
        "https://www.deloitte.com/ng/en/services/risk-advisory/services/how-artificial-intelligence-is-transforming-the-financial-services-industry.html",
        "https://mitsloan.mit.edu/ideas-made-to-matter/financial-services-deliberate-approach-to-ai",
        "https://online.mason.wm.edu/blog/the-future-of-finance-ai-machine-learning-predictive-analytics",
        "https://www.imf.org/en/News/Articles/2024/09/06/sp090624-artificial-intelligence-and-its-impact-on-financial-markets-and-financial-stability",
        "https://www.chicagobooth.edu/review/evolution-ai-finance",
        "https://builtin.com/artificial-intelligence/ai-finance-banking-applications-companies",
        "https://www.forbes.com/sites/kathleenwalch/2024/09/14/how-ai-is-transforming-the-finance-industry/",
        "https://www.ey.com/en_gr/insights/financial-services/how-artificial-intelligence-is-reshaping-the-financial-services-industry",
        "https://www.fsb.org/2024/11/the-financial-stability-implications-of-artificial-intelligence/",
        "https://slalom.com/us/en/insights/financial-services-outlook-2025?utm_source=google&utm_medium=paid_search&utm_campaign=2025-Q1-GBL-IND-CON-Paid-Ads-FS-Outlook&utm_term=Technology&utm_content=FS_Outlook&creative=730300919730&keyword=artificial%20intelligence%20in%20financial%20services&matchtype=p&network=g&device=c&gad_source=1&gclid=Cj0KCQiAz6q-BhCfARIsAOezPxlaPpmzH6jUKQWhrDcBo6v38syDCQGI_OTqCgiWlXC53RbDiPszH4MaAuacEALw_wcB",

        # algorithmic trading
        "https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp",
        "https://corporatefinanceinstitute.com/resources/equities/algorithmic-trading/",
        "https://en.wikipedia.org/wiki/Algorithmic_trading",
        "https://www.stonex.com/en/financial-glossary/algorithmic-trading/",
        "https://www.fool.com/terms/a/algorithmic-trading/",
        "https://www.utradealgos.com/blog/how-to-get-started-with-algo-trading-a-step-by-step-guide",
        "https://www.velvetech.com/blog/high-frequency-algorithmic-trading/",
        "https://www.angelone.in/knowledge-center/online-share-trading/what-is-algo-trading",
        "https://www.rsj.com/en/securities/algorithmic-trading.html",
        "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://faculty.haas.berkeley.edu/hender/Algo.pdf",
        "https://www.investopedia.com/terms/a/algorithmictrading.asp",
        "https://www.fe.training/free-resources/capital-markets/what-is-algorithmic-trading/"

        # risk in defi
        "https://medium.com/intotheblock/four-levels-of-risk-management-in-defi-2a1173465a46",
        "https://streamflow.finance/blog/risk-management-defi/", # pdf
        "https://www.mdpi.com/1911-8074/18/1/38",
        "https://foundershield.com/blog/guide-to-defi-risk-management/",
        # "https://trakx.io/resources/insights/risk-management-in-crypto-trading-effective-guide/",
        "https://blog.shimmer.network/beginners-guide-risk-management-1/",
        "https://www.nuant.com/blog/how-to-measure-and-manage-defi-risks",
        "https://tokenminds.co/blog/knowledge-base/defi-risk-management",
        "https://www.forbes.com/councils/forbesbusinesscouncil/2025/03/06/decentralized-finance-and-fraud-navigating-security-risks/",
        "https://www.linkedin.com/advice/3/what-risks-defi-how-can-you-avoid-them-skills-blockchain-n3avf",
        "https://www2.deloitte.com/us/en/pages/audit/articles/blockchain-digital-assets-risk-management.html",
        "https://www.nadcab.com/blog/smart-contract-risks",
        # "https://www.weforum.org/stories/2024/07/smart-contracts-technology-cybersecurity-legal-risks/",
        "https://synodus.com/blog/fintech/defi-risk/",
        # "https://pmc.ncbi.nlm.nih.gov/articles/PMC10088710/",
        "https://arxiv.org/html/2312.01018v1",
        # "https://www.researchgate.net/publication/384258147_Risk_Management_in_DeFi_copy",   
        "https://www.rapidinnovation.io/post/smart-contracts-and-defi-transforming-decentralized-finance",
        "https://alphadevelopment.com/insights/decentralised-finance-defi-risks-how-to-protect-your-investments-and-navigate-the-crypto-landscape-safely/",
        "https://www.futurelearn.com/info/courses/defi-exploring-decentralised-finance-with-blockchain-technologies/0/steps/256218"
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