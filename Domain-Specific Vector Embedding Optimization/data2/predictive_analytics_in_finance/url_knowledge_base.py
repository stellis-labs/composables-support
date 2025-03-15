# url_knowledge_base.py
import requests
import logging
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from llama_index.core import Document
from clean_text import CleanText

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class URLKnowledgeBase:
    """Handles fetching and processing knowledge from URLs."""
    
    def __init__(self, knowledge_links):
        self.knowledge_links = knowledge_links
        self.documents = []

    def fetch_documents(self):
        """Fetch text from URLs concurrently."""
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(self.process_url, self.knowledge_links)

        self.documents = list(filter(None, results))
        logger.info(f"Loaded {len(self.documents)} documents from URLs.")

    def extract_text_from_html(self, html):
        """Extract text from an HTML string using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        # remove unwanted tags
        for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # extract cleaned text
        cleantextclass = CleanText(text)
        return cleantextclass.clean_text_function()

    def process_url(self, url):
        """Fetch and process text from a single URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            main_text = self.extract_text_from_html(response.content)
            cleaned_text1 = re.sub(r'\s+', ' ', main_text) # removes extra whitespace
            cleaned_text2 = re.sub(r'[^a-zA-Z0-9,.!?%$@#&() -]','',cleaned_text1)
            text = cleaned_text2.strip() # removes leading/trailing whitespace     
            logger.info(f"Successfully fetched {url}")
            return Document(text=text, metadata={"source": url})
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load document from {url}: {e}")
            return None
