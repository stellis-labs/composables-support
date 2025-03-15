import os
import logging
import pdfplumber
from llama_index.core import Document

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFKnowledgeBase:
    """Handles extracting and processing knowledge from PDFs."""
    
    def __init__(self, pdf_folder):
        self.pdf_folder = pdf_folder
        self.documents = []

    def extract_documents(self):
        """Extract text from all PDFs in the folder."""
        if not os.path.exists(self.pdf_folder):
            logger.warning(f"PDF folder {self.pdf_folder} does not exist.")
            return

        pdf_files = [os.path.join(self.pdf_folder, file) for file in os.listdir(self.pdf_folder) if file.endswith(".pdf")]
        if not pdf_files:
            logger.warning("No PDFs found in the folder.")
            return

        for pdf_path in pdf_files:
            try:
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    document = Document(text=text, metadata={"source": pdf_path})
                    self.documents.append(document)
                    logger.info(f"Loaded document from {pdf_path}")
                else:
                    logger.warning(f"No text extracted from {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return None
