import os
import time
import random
import re
import hashlib
import json
import shutil
import requests
from typing import List, Dict, Union, Set, Tuple
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# Vector store and embedding components
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class RateLimiter:
    """
    Comprehensive rate limiter that manages all Groq API rate limit types:
    - Requests per minute (RPM)
    - Requests per day (RPD)
    - Tokens per minute (TPM)
    - Tokens per day (TPD)
    
    This class tracks usage across different time windows and proactively
    manages all limit types to prevent hitting any rate limits.
    """
    
    def __init__(self, limits=None):
        """
        Initialize rate limiter with all limit types.
        
        Args:
            limits: Dict containing rate limits (defaults to reasonable values if not provided)
        """
        # Default conservative limits
        self.limits = limits or {
            "rpm": 45,          # Requests per minute (default: 45)
            "rpd": 10000,       # Requests per day (default: 10000)
            "tpm": 5500,        # Tokens per minute (default: 5500, below 6000 for safety)
            "tpd": 550000       # Tokens per day (default: 550000, below 600000 for safety)
        }
        
        # Counters for usage tracking
        self.requests_minute = 0
        self.requests_day = 0
        self.tokens_minute = 0
        self.tokens_day = 0
        
        # Timestamps for window tracking
        self.minute_window_start = time.time()
        self.day_window_start = time.time()
        self.last_request_time = 0
        
        # Constants
        self.MINUTE_SECONDS = 60
        self.DAY_SECONDS = 86400  # 24 hours in seconds
    
    def reset_if_needed(self):
        """Reset counters if we've moved to a new time window."""
        current_time = time.time()
        
        # Reset minute counters if minute has passed
        if current_time - self.minute_window_start >= self.MINUTE_SECONDS:
            self.requests_minute = 0
            self.tokens_minute = 0
            self.minute_window_start = current_time
        
        # Reset day counters if day has passed
        if current_time - self.day_window_start >= self.DAY_SECONDS:
            self.requests_day = 0
            self.tokens_day = 0
            self.day_window_start = current_time
    
    def add_usage(self, requests=1, tokens=0):
        """
        Add usage statistics after making a request.
        
        Args:
            requests: Number of requests made (default: 1)
            tokens: Number of tokens used in the request
        """
        self.reset_if_needed()
        
        # Update counters
        self.requests_minute += requests
        self.requests_day += requests
        self.tokens_minute += tokens
        self.tokens_day += tokens
        self.last_request_time = time.time()
    
    def wait_if_needed(self, estimated_tokens=0):
        """
        Check all rate limits and wait if necessary before making a request.
        
        Args:
            estimated_tokens: Expected token usage for upcoming request
            
        Returns:
            bool: True if wait was needed, False otherwise
        """
        self.reset_if_needed()
        current_time = time.time()
        
        # Ensure minimum gap between requests (rate smoothing)
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 1.0:  # Minimum 1 second between requests
            time.sleep(1.0 - time_since_last_request)
            current_time = time.time()
            
        # Check which limit will be hit first and calculate wait time
        wait_time = 0
        limit_type = None
        
        # Check minute-based limits
        time_in_minute = current_time - self.minute_window_start
        time_to_reset_minute = max(0, self.MINUTE_SECONDS - time_in_minute)
        
        # Check RPM limit
        if self.requests_minute + 1 >= self.limits["rpm"]:
            wait_time = time_to_reset_minute + 1  # Add buffer second
            limit_type = "RPM"
        
        # Check TPM limit
        if self.tokens_minute + estimated_tokens >= self.limits["tpm"]:
            if wait_time == 0 or time_to_reset_minute + 1 < wait_time:
                wait_time = time_to_reset_minute + 1
                limit_type = "TPM"
        
        # Check day-based limits (only warn, as waiting for a day is impractical)
        # RPD limit check
        if self.requests_day + 1 >= self.limits["rpd"]:
            print(f"WARNING: Approaching requests per day limit! ({self.requests_day}/{self.limits['rpd']})")
            limit_type = "RPD"
        
        # TPD limit check  
        if self.tokens_day + estimated_tokens >= self.limits["tpd"]:
            print(f"WARNING: Approaching tokens per day limit! ({self.tokens_day}/{self.limits['tpd']})")
            limit_type = "TPD"
        
        # Wait if necessary
        if wait_time > 0:
            print(f"Approaching {limit_type} limit. Waiting {wait_time:.2f} seconds before continuing...")
            time.sleep(wait_time)
            
            # Reset appropriate counters after waiting
            if limit_type in ["RPM", "TPM"]:
                self.requests_minute = 0
                self.tokens_minute = 0
                self.minute_window_start = time.time()
            
            return True
        
        return False
    
    def get_usage_stats(self):
        """Return current usage statistics."""
        self.reset_if_needed()
        return {
            "requests_minute": self.requests_minute,
            "requests_day": self.requests_day,
            "tokens_minute": self.tokens_minute,
            "tokens_day": self.tokens_day,
            "rpm_percent": (self.requests_minute / self.limits["rpm"]) * 100,
            "rpd_percent": (self.requests_day / self.limits["rpd"]) * 100,
            "tpm_percent": (self.tokens_minute / self.limits["tpm"]) * 100,
            "tpd_percent": (self.tokens_day / self.limits["tpd"]) * 100
        }


class FinancialDatasetGenerator:
    """
    Generate synthetic financial domain datasets for vector embedding fine-tuning.
    
    This class loads financial knowledge, creates vector representations,
    generates diverse questions with varying complexity, and produces
    synthetic question-answer pairs optimized for embedding training.
    """
    
    def __init__(
        self,
        knowledge_sources: Dict[str, Union[List[str], str]],
        seed_questions: List[str],
        api_key: str,
        llm_model: str = "llama3-70b-8192",
        max_response_tokens: int = 500,
        max_question_words: int = 55,
        allow_complex_questions: bool = True
    ):
        """
        Initialize the dataset generator with enhanced control over question complexity.
        
        Args:
            knowledge_sources: Dict with keys 'urls' and 'pdf_dir'
            seed_questions: List of questions or empty list (loaded from prompt.txt if empty)
            api_key: Groq API key
            llm_model: Name of the LLM model to use
            max_response_tokens: Maximum tokens per response
            max_question_words: Base maximum for question words (simple questions)
            allow_complex_questions: Whether to allow more complex, multi-part questions
        """
        self.knowledge_sources = knowledge_sources
        self.seed_questions = seed_questions
        self.documents = []
        self.api_key = api_key
        self.llm_model = llm_model
        self.chroma_dir = "./chroma_db"
        self.max_response_tokens = max_response_tokens
        self.max_question_words = max_question_words
        self.allow_complex_questions = allow_complex_questions
        
        # Define rate limits based on Groq's typical limits
        self.rate_limits = {
            "rpm": 45,        # Requests per minute
            "rpd": 10000,     # Requests per day
            "tpm": 5500,      # Tokens per minute (conservative from 6000)
            "tpd": 550000     # Tokens per day (conservative from 600000)
        }
        
        # Question complexity settings
        self.complexity_types = {
            "basic": {"weight": 0.5, "max_words": max_question_words},  # 50% simple questions
            "relationship": {"weight": 0.2, "max_words": max_question_words * 1.2},  # 20% relationship questions
            "application": {"weight": 0.2, "max_words": max_question_words * 1.3},  # 20% application questions
            "multi_part": {"weight": 0.1, "max_words": max_question_words * 1.5}    # 10% complex questions
        }
        
        # Question tracking
        self.generated_questions_hash = set()
        self.question_categories = {
            "factual": 0,
            "conceptual": 0, 
            "comparative": 0,
            "hypothetical": 0,
            "procedural": 0,
            "evaluative": 0,
            "technical": 0,
            "application": 0
        }
        
        # Complexity tracking
        self.complexity_counts = {c_type: 0 for c_type in self.complexity_types.keys()}
        
        # Enhanced rate limiter
        self.rate_limiter = RateLimiter(limits=self.rate_limits)
        
        # Initialize components
        self.initialize_components()
    
    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================
    
    def initialize_components(self):
        """Initialize and configure all necessary components."""
        try:
            print("Setting up components...")
            # Clean existing Chroma directory to avoid dimension mismatch
            if os.path.exists(self.chroma_dir):
                shutil.rmtree(self.chroma_dir)
                print("Removed existing Chroma database")
            
            # Set up embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                embed_batch_size=4,
                max_length=256
            )
            
            # Set up LLM via Groq
            self.llm = Groq(
                model=self.llm_model,
                api_key=self.api_key
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 256
            Settings.chunk_overlap = 20
            
            # Load knowledge sources
            print("Loading knowledge sources...")
            self.load_documents_from_urls()
            self.load_documents_from_pdfs()
            
            if not self.documents:
                raise ValueError("No documents were successfully loaded")
            
            # Extract key concepts from documents
            self.key_concepts = self.extract_key_concepts()
            
            # Set up vector store
            print("Setting up vector store...")
            self.setup_vector_store()
            
            print("Initialization complete")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")
    
    def setup_vector_store(self):
        """Set up the vector store and index."""
        try:
            # Clean and prepare documents
            cleaned_docs = []
            for doc in self.documents:
                try:
                    cleaned_text = self.clean_text(doc.text)
                    if cleaned_text.strip():
                        cleaned_docs.append(Document(
                            text=cleaned_text,
                            metadata=doc.metadata
                        ))
                except Exception as e:
                    print(f"Error cleaning document: {str(e)}")
                    continue

            if not cleaned_docs:
                raise ValueError("No valid documents after cleaning")

            # Create Chroma collection
            chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            collection_name = f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Create vector store and index
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.index = VectorStoreIndex.from_documents(
                cleaned_docs,
                storage_context=storage_context,
                show_progress=True
            )
            
            print(f"Created vector index with {len(cleaned_docs)} documents")
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup vector store: {str(e)}")
    
    # =========================================================================
    # DOCUMENT LOADING METHODS
    # =========================================================================
    
    def load_documents_from_urls(self):
        """Load and process documents from URLs."""
        if 'urls' not in self.knowledge_sources or not self.knowledge_sources['urls']:
            print("No URLs provided in knowledge sources")
            return
            
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for url in self.knowledge_sources['urls']:
            try:
                print(f"Loading content from {url}")
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove non-content elements
                for element in soup(['script', 'style', 'footer', 'nav', 'header']):
                    element.decompose()
                
                # Extract paragraphs and other content elements
                paragraphs = soup.find_all(['p', 'article', 'section'])
                text = ' '.join(p.get_text(strip=True) for p in paragraphs)
                
                if not text.strip():
                    print(f"Warning: No text content extracted from {url}")
                    continue
                    
                # Create document
                document = Document(
                    text=self.clean_text(text),
                    metadata={
                        "source": url,
                        "type": "url",
                        "title": soup.title.string if soup.title else "Untitled"
                    }
                )
                self.documents.append(document)
                print(f"Successfully loaded document from URL: {url}")
                
            except Exception as e:
                print(f"Error loading URL {url}: {str(e)}")

    def load_documents_from_pdfs(self):
        """Load and process documents from PDF files."""
        if 'pdf_dir' not in self.knowledge_sources or not self.knowledge_sources['pdf_dir']:
            print("No PDF directory provided in knowledge sources")
            return
            
        pdf_dir = self.knowledge_sources['pdf_dir']
        if not os.path.exists(pdf_dir):
            print(f"PDF directory {pdf_dir} does not exist")
            return

        for filename in os.listdir(pdf_dir):
            if not filename.endswith('.pdf'):
                continue
                
            file_path = os.path.join(pdf_dir, filename)
            try:
                print(f"Loading PDF: {filename}")
                reader = PdfReader(file_path)
                text_chunks = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_chunks.append(page_text)
                    except Exception as e:
                        print(f"Error extracting text from page {page_num} in {filename}: {str(e)}")
                
                if not text_chunks:
                    print(f"Warning: No text extracted from {filename}")
                    continue
                
                # Create document
                full_text = self.clean_text(' '.join(text_chunks))
                document = Document(
                    text=full_text,
                    metadata={
                        "file_name": filename,
                        "source": file_path,
                        "type": "pdf",
                        "pages": len(reader.pages)
                    }
                )
                self.documents.append(document)
                print(f"Successfully loaded PDF document: {filename}")
                
            except Exception as e:
                print(f"Error processing PDF file {filename}: {str(e)}")

    def load_seed_questions(self, file_path: str) -> List[str]:
        """Load seed questions from a text file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                seed_questions = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(seed_questions)} seed questions from {file_path}")
            return seed_questions
        except Exception as e:
            raise RuntimeError(f"Failed to load seed questions from {file_path}: {str(e)}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding."""
        # Normalize whitespace
        text = ' '.join(text.split())
        # Truncate if too long
        return text[:5000] if len(text) > 5000 else text
    
    def extract_key_concepts(self) -> List[str]:
        """Extract key domain-specific concepts from the loaded documents and seed questions."""
        try:
            # Prepare sample text from documents and seed questions
            sample_text = ""
            # Sample from documents
            for doc in self.documents[:5]:
                sample_text += doc.text[:1000] + "\n\n"
            
            # Include seed questions
            if self.seed_questions:
                sample_text += "\n\nSEED QUESTIONS:\n"
                for q in self.seed_questions[:20]:
                    sample_text += f"- {q}\n"
            
            # Create finance-specific prompt for concept extraction
            extract_prompt = f"""
            You are an expert in financial machine learning and information retrieval.
            Based on the following text from financial documents and seed questions, identify 15-20 key technical concepts 
            that are central to the financial domain.
            
            Focus on extracting concepts from these categories:
            1. Financial instruments and markets (e.g., mutual funds, derivatives)
            2. Financial analysis techniques (e.g., portfolio theory, risk assessment)
            3. Financial technologies (e.g., blockchain, algorithmic trading)
            4. AI/ML applications in finance (e.g., transformers for prediction)
            
            TEXT AND QUESTIONS:
            {sample_text}

            Return only a comma-separated list of key financial concepts, with no additional text.
            """
            
            # Wait if needed based on token estimates
            estimated_tokens = len(extract_prompt.split()) * 2
            self.rate_limiter.wait_if_needed(estimated_tokens)
            
            # Execute with rate limiting
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.llm.complete(extract_prompt)
                    # Track tokens and usage
                    response_tokens = len(str(response).split())
                    self.rate_limiter.add_usage(requests=1, tokens=estimated_tokens + response_tokens)
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Error extracting concepts, retrying: {str(e)}")
                        time.sleep(5)
                    else:
                        raise
            
            # Process response
            concepts = [concept.strip() for concept in str(response).split(',')]
            print(f"Extracted {len(concepts)} key financial concepts")
            return concepts
            
        except Exception as e:
            print(f"Warning: Failed to extract key concepts: {str(e)}")
            # Fallback concepts derived from common financial topics
            return [
                "mutual funds", "stock prediction", "credit risk scoring", 
                "blockchain technology", "corporate finance", "agentic AI systems", 
                "financial modeling", "algorithmic trading", "portfolio optimization", 
                "time series analysis", "capital budgeting", "financial engineering"
            ]
    
    def extract_financial_topics(self) -> List[str]:
        """Extract financial topics from seed questions to ensure relevance."""
        # Start with common financial topics
        topics = [
            "mutual funds", "investment", "portfolio", "stock market", "stock price", 
            "algorithmic trading", "credit risk", "financial modeling", "corporate finance",
            "blockchain", "bitcoin", "cryptocurrency", "financial analysis", "risk management",
            "quantitative finance", "capital budgeting", "asset allocation", "bond pricing",
            "options", "derivatives", "financial engineering", "time series", "prediction",
            "fintech", "personal finance", "capital markets", "financial services"
        ]
        
        # Add topics from seed questions
        for q in self.seed_questions:
            # Extract meaningful words that might be topics
            for word in q.split():
                if word.lower() not in ["what", "how", "why", "is", "are", "do", "does", "can", "could", "would", "the", "a", "an"]:
                    if len(word) > 4:  # Only consider words with 5+ characters
                        topics.append(word.lower())
        
        # Remove duplicates
        return list(set(topics))
    
    def determine_question_complexity(self, question: str) -> str:
        """
        Determine the complexity level of a question.
        
        Args:
            question: The question to analyze
            
        Returns:
            str: The complexity level (basic, relationship, application, or multi_part)
        """
        word_count = len(question.split())
        
        # Check for multi-part questions
        multi_part_indicators = [
            " and ", ", and ", "; ", "? ", 
            " part ", " first ", " second ", " third ",
            "compare", "contrast", "relationship between"
        ]
        
        is_multi_part = any(indicator in question.lower() for indicator in multi_part_indicators)
        
        # Check for application questions
        application_indicators = [
            "application", "apply", "implement", "use", "utilized", 
            "employed", "practical", "real-world", "business", "industry"
        ]
        
        is_application = any(indicator in question.lower() for indicator in application_indicators)
        
        # Check for relationship questions
        relationship_indicators = [
            "relate", "connection", "relationship", "between", "impact", 
            "effect", "influence", "correlation", "link", "association"
        ]
        
        is_relationship = any(indicator in question.lower() for indicator in relationship_indicators)
        
        # Determine complexity based on indicators and word count
        if is_multi_part and word_count > self.max_question_words:
            return "multi_part"
        elif is_application:
            return "application"
        elif is_relationship:
            return "relationship"
        else:
            return "basic"
    
    def is_question_within_limit(self, question: str, complexity_type: str = None) -> bool:
        """
        Check if a question is within the allowed word limit for its complexity.
        
        Args:
            question: The question to check
            complexity_type: The question's complexity type (if known)
            
        Returns:
            bool: True if within limit, False otherwise
        """
        word_count = len(question.split())
        
        # Determine complexity if not provided
        if not complexity_type:
            complexity_type = self.determine_question_complexity(question)
            
        # Get word limit for this complexity type
        max_words = self.complexity_types[complexity_type]["max_words"]
        
        return word_count <= max_words
    
    def simplify_question(self, question: str) -> Tuple[str, str, bool]:
        """
        Simplify a question if it exceeds the maximum allowed words for its complexity.
        
        Args:
            question: The question to simplify
            
        Returns:
            Tuple[str, str, bool]: (simplified question, complexity type, was simplified)
        """
        # Determine complexity and check if simplification is needed
        complexity_type = self.determine_question_complexity(question)
        max_words = self.complexity_types[complexity_type]["max_words"]
        
        if len(question.split()) <= max_words:
            return question, complexity_type, False
            
        # First attempt: Remove introductory phrases
        introductory_phrases = [
            "According to the knowledge base, ",
            "Based on the financial literature, ",
            "In the context of financial markets, ",
            "From the perspective of investment theory, ",
            "With reference to Bogle's investment philosophy, ",
            "Considering the principles of corporate finance, ",
            "As described in the corporate finance overview, ",
            "According to the blockchain report, "
        ]
        
        simplified = question
        for phrase in introductory_phrases:
            if question.startswith(phrase):
                simplified = question.replace(phrase, "", 1)
                if len(simplified.split()) <= max_words:
                    return simplified, complexity_type, True
        
        # For multi-part questions, try to preserve structure while simplifying
        if complexity_type == "multi_part":
            # Try to split by common separators
            parts = []
            remaining = question
            
            for separator in [", and ", " and ", "; "]:
                if separator in remaining:
                    parts.extend(remaining.split(separator))
                    break
            
            # If we found parts, simplify each part
            if len(parts) > 1:
                # Set reasonable word limit per part
                words_per_part = max_words // len(parts)
                
                simplified_parts = []
                for part in parts:
                    # Simplify each part
                    part_words = part.split()
                    if len(part_words) > words_per_part:
                        simplified_part = " ".join(part_words[:words_per_part])
                        # Ensure it ends with a question mark if it's the last part
                        if part == parts[-1] and not simplified_part.endswith("?"):
                            simplified_part += "?"
                        simplified_parts.append(simplified_part)
                    else:
                        simplified_parts.append(part)
                
                # Recombine parts
                simplified = " and ".join(simplified_parts)
                if not simplified.endswith("?"):
                    simplified += "?"
                
                if len(simplified.split()) <= max_words:
                    return simplified, complexity_type, True
        
        # If the above didn't work, try to preserve the core question
        words = question.split()
        question_indicators = ["what", "how", "why", "when", "where", "which", "who", "can", "does", "is", "are"]
        
        # Check if question starts with an indicator
        starts_with_indicator = words[0].lower() in question_indicators
        
        if starts_with_indicator:
            # Preserve the question structure while shortening
            simplified = " ".join(words[:max_words])
            
            # Ensure we end with a question mark
            if not simplified.endswith("?"):
                simplified += "?"
                
            return simplified, complexity_type, True
        
        # Last resort: just truncate to max words
        simplified = " ".join(words[:max_words])
        if not simplified.endswith("?"):
            simplified += "?"
            
        return simplified, complexity_type, True
    
    def truncate_response(self, text: str) -> str:
        """
        Truncate response to stay within token limit while ensuring complete sentences.
        
        Args:
            text: The response text to truncate
            
        Returns:
            str: Truncated text ending at a natural sentence boundary
        """
        words = text.split()
        
        if len(words) <= self.max_response_tokens:
            return text
            
        # Keep approximately max_tokens words
        truncated = " ".join(words[:self.max_response_tokens])
        
        # Find the last complete sentence (look for various sentence endings)
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        end_position = -1
        
        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > end_position:
                end_position = pos
        
        # If we found a sentence boundary and it's not too short
        min_content_threshold = 0.6  # Keep at least 60% of the max tokens
        if end_position > 0 and end_position > min_content_threshold * len(truncated):
            # Include the period/question mark/exclamation point in the truncation
            return truncated[:end_position + 1].strip()
        
        # If no good sentence boundary found, try to end at a phrase boundary
        phrase_endings = [", ", "; ", "—", ":\n", ";\n"]
        end_position = -1
        
        for ending in phrase_endings:
            pos = truncated.rfind(ending)
            if pos > end_position and pos > 0.8 * len(truncated):  # Only if near the end
                end_position = pos
        
        if end_position > 0:
            # Include the punctuation in the truncated text
            return truncated[:end_position + 1].strip() + "..."
        
        # Last resort: Just truncate at token limit and add ellipsis
        return truncated.strip() + "..."
    
    def classify_financial_subtopic(self, question: str) -> str:
        """Classify the financial subtopic of a question."""
        q_lower = question.lower()
        
        subtopic_keywords = {
            "investments": ["mutual fund", "portfolio", "investment", "asset", "stock", "security", "bogle"],
            "trading": ["algorithmic", "trading", "strategy", "market", "position", "order", "execution"],
            "risk_management": ["risk", "volatility", "exposure", "hedge", "diversification"],
            "banking": ["bank", "loan", "credit", "lending", "deposit"],
            "fintech": ["fintech", "digital", "app", "platform", "technology"],
            "corporate_finance": ["corporate", "capital", "budget", "merger", "acquisition", "dividend"],
            "crypto_blockchain": ["blockchain", "crypto", "bitcoin", "token", "ledger", "smart contract"],
            "ml_finance": ["machine learning", "ai", "model", "algorithm", "prediction", "transformer"],
            "quantitative_finance": ["quant", "mathematical", "statistical", "stochastic", "calculus"]
        }
        
        for subtopic, keywords in subtopic_keywords.items():
            if any(keyword in q_lower for keyword in keywords):
                return subtopic
                
        return "general_finance"
    
    def is_finance_relevant(self, question: str, financial_topics: List[str]) -> bool:
        """Check if a question is relevant to finance domain."""
        q_lower = question.lower()
        
        # Check for finance relevance
        finance_relevant = any(topic in q_lower for topic in financial_topics)
        
        # If not directly relevant, check for indirect relevance
        if not finance_relevant:
            relationship_indicators = ["similar", "related", "compared to", "relationship", 
                                      "correlation", "pattern", "trend", "cluster"]
            relationship_relevant = any(indicator in q_lower for indicator in relationship_indicators)
            
            finance_relevant = relationship_relevant
        
        return finance_relevant
    
    # =========================================================================
    # PROMPT TEMPLATE METHODS
    # =========================================================================
    
    def get_question_type_guidance(self, question_type: str) -> str:
        """Get specific guidance for different question types in the financial domain."""
        guidance = {
            "factual": "ask for specific information, definitions, techniques, or methods related to financial concepts from the knowledge base.",
            "conceptual": "explore theoretical aspects, principles, and abstract concepts behind financial topics such as mutual funds, algorithmic trading, or blockchain technology.",
            "comparative": "contrast different approaches, methods, or models covered in the knowledge base (e.g., comparing traditional vs. AI methods for credit scoring).",
            "hypothetical": "explore potential scenarios, future developments, or theoretical applications of financial concepts (e.g., how blockchain might transform financial intermediation).",
            "procedural": "focus on steps, processes, implementations, or methodologies covered in the knowledge base (e.g., how to build financial models or implement trading strategies).",
            "evaluative": "assess the effectiveness, challenges, limitations, or benefits of financial approaches mentioned in the knowledge base.",
            "technical": "focus on the technical implementation details related to financial technologies (e.g., transformer architecture for stock prediction, Python for finance).",
            "application": "explore specific applications of financial concepts in real-world contexts (e.g., fraud detection, market prediction, risk assessment)."
        }
        return guidance.get(question_type, "focus on relevant financial topics from the knowledge base")
    
    def create_prompt_template(self, question_type: str, batch_num: int) -> str:
        """
        Create a prompt template for generating financial domain questions with varied complexity.
        
        Args:
            question_type: The type of questions to generate (factual, conceptual, etc.)
            batch_num: The batch number for this generation round
            
        Returns:
            str: A prompt template for the LLM
        """
        # Load seed questions from prompt.txt if not provided
        if not self.seed_questions:
            self.seed_questions = self.load_seed_questions("prompt.txt")
        
        # Categorize seed questions by topic
        finance_categories = {
            "investments": [],
            "ml_finance": [],
            "blockchain": [],
            "corporate_finance": [],
            "algorithmic_trading": [],
            "financial_modeling": []
        }
        
        # Simple keyword-based categorization
        for q in self.seed_questions:
            if any(kw in q.lower() for kw in ["mutual fund", "bogle", "portfolio", "investing"]):
                finance_categories["investments"].append(q)
            elif any(kw in q.lower() for kw in ["transformer", "ai", "model", "python", "algorithm"]):
                finance_categories["ml_finance"].append(q)
            elif any(kw in q.lower() for kw in ["blockchain", "bitcoin", "crypto"]):
                finance_categories["blockchain"].append(q)
            elif any(kw in q.lower() for kw in ["corporate", "cfo", "capital", "budget"]):
                finance_categories["corporate_finance"].append(q)
            elif any(kw in q.lower() for kw in ["algorithmic", "trading", "strategy", "backtesting"]):
                finance_categories["algorithmic_trading"].append(q)
            elif any(kw in q.lower() for kw in ["model", "financial model", "pricing", "valuation"]):
                finance_categories["financial_modeling"].append(q)
        
        # Select 1-2 questions from 2 categories to save tokens
        categories_to_sample = random.sample(list(finance_categories.keys()), min(2, len(finance_categories)))
        selected_seeds = []
        
        for category in categories_to_sample:
            category_questions = finance_categories[category]
            if category_questions:
                sample_size = min(2, len(category_questions))
                selected_seeds.extend(random.sample(category_questions, sample_size))
        
        # If we didn't get enough, add random samples
        if len(selected_seeds) < 3:
            additional_samples = random.sample(self.seed_questions, min(3, len(self.seed_questions)))
            # Remove duplicates
            additional_samples = [q for q in additional_samples if q not in selected_seeds]
            selected_seeds.extend(additional_samples[:3-len(selected_seeds)])
        
        # Select financial topics for focus
        financial_topics = [
            "mutual fund strategies", "investment principles", "stock price prediction",
            "credit risk assessment", "blockchain technology", "corporate finance",
            "agentic AI systems", "financial modeling", "algorithmic trading",
            "quantitative finance", "capital budgeting", "financial engineering"
        ]
        
        # Select topics to focus on
        selected_topics = random.sample(financial_topics, min(3, len(financial_topics)))
        
        # Determine which complexity types to focus on for this batch
        # Adjust weights based on existing distribution to ensure balance
        adjusted_weights = {}
        total_questions = sum(self.complexity_counts.values()) or 1  # Avoid division by zero
        
        for ctype, settings in self.complexity_types.items():
            # Calculate the current proportion of this complexity type
            current_proportion = self.complexity_counts.get(ctype, 0) / total_questions
            # Adjust weight - increase if underrepresented, decrease if overrepresented
            target_proportion = settings["weight"]
            adjustment_factor = 1.5 if current_proportion < target_proportion else 0.7
            adjusted_weights[ctype] = settings["weight"] * adjustment_factor
        
        # Normalize weights to sum to 1
        weight_sum = sum(adjusted_weights.values())
        normalized_weights = {ctype: w/weight_sum for ctype, w in adjusted_weights.items()}
        
        # Select complexity types to emphasize in this batch
        complexity_emphasis = random.choices(
            list(self.complexity_types.keys()),
            weights=[normalized_weights[ctype] for ctype in self.complexity_types.keys()],
            k=2  # Select 2 complexity types to emphasize
        )
        
        # Examples for each complexity level
        complexity_examples = {
            "basic": [
                "What key elements of mutual fund selection does Bogle describe?",
                "How is the transformer-based deep learning model structured for stock price prediction?"
            ],
            "relationship": [
                "How does Bogle's concept of cost efficiency relate to long-term wealth creation in mutual fund investing?",
                "What is the relationship between time2vec encoding and transformer model effectiveness for financial time series data?"
            ],
            "application": [
                "How can the agentic AI systems described in the knowledge base be applied to improve model risk management in real-world financial institutions?",
                "What practical applications of blockchain technology in financial intermediation would address the regulatory challenges mentioned in the report?"
            ],
            "multi_part": [
                "What are the primary differences between traditional and AI-based credit risk scoring methods, and how do these differences impact SMEs' access to credit?",
                "How does corporate finance explain the agency problem between managers and shareholders, and what strategies are recommended for mitigating these conflicts in large corporations?"
            ]
        }
        
        # Get examples for the emphasized complexity types
        selected_examples = []
        for complexity in complexity_emphasis:
            selected_examples.extend(complexity_examples[complexity])
        
        # Ensure we have at least 2 examples
        if len(selected_examples) < 2:
            # Add examples from random complexity types
            additional_types = [ct for ct in self.complexity_types.keys() if ct not in complexity_emphasis]
            if additional_types:
                additional_type = random.choice(additional_types)
                selected_examples.extend(complexity_examples[additional_type])
        
        # Limit to 2 examples to keep prompt focused
        selected_examples = selected_examples[:2]
        
        # Create comprehensive complexity guidance
        complexity_guidance = "\n".join([
            f"- BASIC: Simple, straightforward questions about a single concept (15-20 words)",
            f"- RELATIONSHIP: Questions exploring connections between concepts (20-40 words)",
            f"- APPLICATION: Questions about applying concepts to real-world scenarios (25-45 words)",
            f"- MULTI-PART: Complex questions with multiple components or related subquestions (40-75 words)"
        ])
        
        # Build focus message for this batch
        batch_topics = ", ".join(selected_topics)
        complexity_focus = ", ".join(complexity_emphasis).upper()
        batch_focus = f"For this batch (#{batch_num}), focus on {question_type} questions about {batch_topics} with emphasis on {complexity_focus} complexity levels."
        
        # Create the complete prompt template with emphasis on variety
        prompt_template = f"""
You are a financial expert. Generate 10 diverse {question_type} questions about the topics below with VARIED COMPLEXITY LEVELS.

SEED QUESTIONS FROM KNOWLEDGE BASE:
{chr(10).join(f"- {q}" for q in selected_seeds)}

FOCUS TOPICS:
{batch_topics}

QUESTION TYPE GUIDANCE:
{question_type.upper()} questions should {self.get_question_type_guidance(question_type)}

COMPLEXITY LEVELS:
{complexity_guidance}

EXAMPLES:
- {selected_examples[0]}
- {selected_examples[1]}

{batch_focus}

REQUIREMENTS:
1. Questions must be directly related to financial topics from the knowledge base
2. Include a balanced mix of complexity levels in your questions
3. For RELATIONSHIP questions, explicitly explore connections between concepts
4. For APPLICATION questions, focus on real-world applications of financial concepts
5. For MULTI-PART questions, include 2-3 related components separated by commas or "and"
6. Vary question length based on complexity (from simple 15-word questions to complex 75-word questions)
7. Avoid unnecessary introductory phrases like "According to the knowledge base"

Return only the questions, one per line, no numbering or additional text.
"""
        return prompt_template
    
    # =========================================================================
    # QUESTION GENERATION METHODS
    # =========================================================================
    
    def generate_prompts(self, num_prompts: int) -> List[Dict]:
        """
        Generate diverse questions for the dataset with varied complexity.
        
        Args:
            num_prompts: The number of questions to generate
            
        Returns:
            List[Dict]: List of generated question items with metadata
        """
        # Load seed questions if needed
        if not self.seed_questions:
            self.seed_questions = self.load_seed_questions("prompt.txt")
        
        # Question type distribution with weights
        weights = {
            "factual": 0.1,       # 10%
            "conceptual": 0.15,   # 15%
            "comparative": 0.15,  # 15%
            "hypothetical": 0.1,  # 10%
            "procedural": 0.15,   # 15%
            "evaluative": 0.1,    # 10%
            "technical": 0.1,     # 10%
            "application": 0.15   # 15%
        }
        
        # Calculate target count for each question type
        question_types = list(self.question_categories.keys())
        type_distribution = {}
        remaining = num_prompts
        
        # First pass: Calculate based on weights
        for qtype, weight in weights.items():
            type_count = int(num_prompts * weight)
            type_distribution[qtype] = type_count
            remaining -= type_count
        
        # Distribute any remainder
        for i in range(remaining):
            type_distribution[question_types[i % len(question_types)]] += 1
            
        print(f"Question type distribution: {type_distribution}")
        
        all_questions = []
        unique_questions_hash = set()
        
        # Get financial topics list for relevance checking
        financial_topics = self.extract_financial_topics()
        
        # Generate questions for each type
        for qtype, target_count in type_distribution.items():
            batch_num = 1
            type_questions = []
            
            while len(type_questions) < target_count:
                try:
                    # Create prompt template with complexity guidance
                    prompt_template = self.create_prompt_template(qtype, batch_num)
                    
                    # Wait if approaching rate limits
                    estimated_tokens = 1100  # Estimate for prompt + response
                    self.rate_limiter.wait_if_needed(estimated_tokens)
                    
                    # Get response with retries for rate limiting
                    max_retries = 5
                    base_delay = 2
                    response = None
                    
                    for retry in range(max_retries):
                        try:
                            response = self.llm.complete(prompt_template)
                            # Add request and token usage
                            response_tokens = len(str(response).split())
                            self.rate_limiter.add_usage(requests=1, tokens=estimated_tokens + response_tokens)
                            break  # Success, exit retry loop
                        except Exception as e:
                            if "rate_limit" in str(e).lower() and retry < max_retries - 1:
                                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                                print(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry+1}/{max_retries})")
                                time.sleep(delay)
                            else:
                                print(f"Error generating questions: {str(e)}")
                                if retry == max_retries - 1:
                                    raise  # Re-raise on last attempt
                    
                    if not response:
                        print(f"Failed to generate questions for batch {batch_num}")
                        continue
                    
                    # Process questions from response
                    batch_questions = [q.strip() for q in str(response).split('\n') if q.strip()]
                    
                    for q in batch_questions:
                        # Ensure question ends with a question mark
                        if not q.endswith('?'):
                            q = q + '?'
                        
                        # Determine complexity and check if simplification is needed
                        original_question = q
                        simplified_question, complexity_type, was_simplified = self.simplify_question(q)
                            
                        # Create a hash to check uniqueness
                        q_hash = hashlib.md5(simplified_question.lower().encode()).hexdigest()
                        
                        # Check if question is relevant to finance domain
                        is_finance_relevant = self.is_finance_relevant(simplified_question, financial_topics)
                        
                        if q_hash not in unique_questions_hash and is_finance_relevant:
                            unique_questions_hash.add(q_hash)
                            
                            # Update complexity count
                            self.complexity_counts[complexity_type] = self.complexity_counts.get(complexity_type, 0) + 1
                            
                            # Classify financial subtopic
                            subtopic = self.classify_financial_subtopic(simplified_question)
                            
                            # Add to questions list with rich metadata
                            type_questions.append({
                                "question": simplified_question,
                                "original_question": original_question if was_simplified else simplified_question,
                                "simplified": was_simplified,
                                "complexity_type": complexity_type,
                                "word_count": len(simplified_question.split()),
                                "type": qtype,
                                "batch": batch_num,
                                "finance_subtopic": subtopic
                            })
                            
                            # Break if we have enough questions
                            if len(type_questions) >= target_count:
                                break
                    
                    # Update progress
                    batch_num += 1
                    print(f"Generated {len(type_questions)}/{target_count} {qtype} questions")
                    
                    # Add delay between batches
                    time.sleep(3)
                    
                    # Avoid infinite loops
                    if batch_num > 15:
                        print(f"Warning: Reached batch limit for {qtype} questions")
                        break
                        
                except Exception as e:
                    print(f"Error during question generation for {qtype}: {str(e)}")
                    time.sleep(3)
            
            # Update question type counts
            self.question_categories[qtype] = len(type_questions)
            all_questions.extend(type_questions)
            
        # Log distribution statistics
        subtopic_distribution = {}
        for q in all_questions:
            subtopic = q.get("finance_subtopic", "other")
            subtopic_distribution[subtopic] = subtopic_distribution.get(subtopic, 0) + 1
            
        complexity_distribution = {k: v for k, v in self.complexity_counts.items()}
            
        print(f"Final question type distribution: {self.question_categories}")
        print(f"Financial subtopic distribution: {subtopic_distribution}")
        print(f"Question complexity distribution: {complexity_distribution}")
        
        return all_questions
    
    # =========================================================================
    # ANSWER GENERATION METHODS
    # =========================================================================
    
    def process_single_prompt(self, prompt_item: Dict) -> Dict:
        """
        Process a single prompt to generate a dataset entry.
        
        Args:
            prompt_item: Dictionary containing question and metadata
            
        Returns:
            Dict: Complete dataset entry with question, answer and metadata
        """
        question = prompt_item["question"]
        original_question = prompt_item.get("original_question", question)
        was_simplified = prompt_item.get("simplified", False)
        complexity_type = prompt_item.get("complexity_type", "basic")
            
        try:
            # Create query engine with settings based on complexity
            similarity_top_k = 3
            response_mode = "compact"
            
            # For more complex questions, use more context
            if complexity_type in ["application", "multi_part"]:
                similarity_top_k = 4
                response_mode = "tree_summarize"
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                response_mode=response_mode
            )
            
            # Craft instruction based on question complexity
            instruction_by_complexity = {
                "basic": "Focus on the key aspects of this question and provide a concise, direct answer.",
                "relationship": "Explore the relationship mentioned in the question, explaining how the concepts connect.",
                "application": "Explain how the concept can be applied in practical contexts, with specific examples if possible.",
                "multi_part": "Address each part of the question separately and thoroughly."
            }
            
            instruction = instruction_by_complexity.get(complexity_type, "Provide a focused answer.")
            
            # Add focused instruction to question
            query_text = f"{question} {instruction}"
            
            # Get response with rate limit handling
            max_retries = 8
            base_delay = 3
            response_text = None
            
            for retry in range(max_retries):
                try:
                    # Wait if needed for rate limiting
                    estimated_query_tokens = len(query_text.split()) * 2 + 300
                    self.rate_limiter.wait_if_needed(estimated_query_tokens)
                    
                    # Execute query
                    response = query_engine.query(query_text)
                    response_text = str(response)
                    
                    # Truncate if response exceeds token limit
                    original_token_count = len(response_text.split())
                    was_truncated = original_token_count > self.max_response_tokens
                    
                    if was_truncated:
                        response_text = self.truncate_response(response_text)
                        print(f"Truncated: {original_token_count} → {len(response_text.split())} tokens")
                    
                    # Track token usage
                    result_tokens = len(response_text.split())
                    self.rate_limiter.add_usage(requests=1, tokens=estimated_query_tokens + result_tokens)
                    
                    # Quality check on response
                    if len(response_text.split()) < 30:
                        print(f"Warning: Response too short, trying alternative approach...")
                        time.sleep(2)
                        
                        # Wait before retry
                        self.rate_limiter.wait_if_needed(estimated_query_tokens)
                        
                        # Try with different settings
                        retry_engine = self.index.as_query_engine(
                            similarity_top_k=similarity_top_k + 1,
                            response_mode="tree_summarize"
                        )
                        response = retry_engine.query(
                            f"{question} Please provide a focused answer addressing all aspects of this question."
                        )
                        response_text = str(response)
                        
                        # Truncate if needed
                        if len(response_text.split()) > self.max_response_tokens:
                            response_text = self.truncate_response(response_text)
                            was_truncated = True
                        
                        # Track token usage for retry
                        retry_tokens = len(response_text.split())
                        self.rate_limiter.add_usage(requests=1, tokens=estimated_query_tokens + retry_tokens)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate_limit" in error_str and retry < max_retries - 1:
                        # Extract wait time if available
                        wait_match = re.search(r"try again in (\d+[.]\d+)s", error_str)
                        if wait_match:
                            suggested_wait = float(wait_match.group(1))
                            delay = suggested_wait + 1  # Add buffer
                        else:
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** retry) + random.uniform(1, 5)
                        
                        print(f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {retry+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        # For non-rate limit errors, try a simpler approach
                        if retry < max_retries - 2:
                            print(f"Error: {str(e)[:100]}... Retrying with simplified settings.")
                            time.sleep(5)
                        else:
                            raise
            
            if not response_text:
                raise ValueError("Failed to get a response after retries")
            
            # Create dataset entry with comprehensive metadata
            token_count = len(response_text.split())
            word_count = len(question.split())
            
            entry = {
                "input": question,
                "output": response_text,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.llm_model,
                    "question_type": prompt_item["type"],
                    "complexity_type": complexity_type,
                    "batch": prompt_item["batch"],
                    "question_words": word_count,
                    "question_simplified": was_simplified,
                    "original_question": original_question if was_simplified else None,
                    "response_tokens": token_count,
                    "total_tokens": word_count + token_count,
                    "truncated": was_truncated,
                    "query_top_k": similarity_top_k,
                    "query_mode": response_mode,
                    "finance_subtopic": prompt_item.get("finance_subtopic", "general_finance")
                }
            }
            
            print(f"Generated: {question[:50]}... ({complexity_type}, {word_count} words, {token_count} tokens)")
            
            # Add delay after successful generation
            time.sleep(1)
            
            return entry
            
        except Exception as e:
            print(f"Error generating answer for: {question[:50]}... Error: {str(e)}")
            time.sleep(3)
            return None
    
    # =========================================================================
    # MAIN DATASET GENERATION METHODS
    # =========================================================================
    
    def generate_dataset(self, num_examples: int = 500) -> List[Dict]:
        """
        Generate a complete synthetic dataset with questions of varied complexity.
        
        Args:
            num_examples: The number of examples to generate
            
        Returns:
            List[Dict]: List of dataset entries
        """
        try:
            print(f"Generating {num_examples} questions across different types and complexity levels...")
            prompt_items = self.generate_prompts(num_examples)
            dataset = []
            
            print(f"Processing {len(prompt_items)} questions to generate answers...")
            
            # Process in small batches with controlled timing
            batch_size = 4  # Small batch size for better rate control
            total_batches = (len(prompt_items) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompt_items))
                batch_prompts = prompt_items[start_idx:end_idx]
                
                print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_prompts)} prompts)")
                
                # Process one at a time for better rate control
                for prompt_item in batch_prompts:
                    # Estimate tokens
                    prompt_length = len(prompt_item["question"].split())
                    estimated_tokens = prompt_length * 2 + 500
                    
                    # Wait if needed
                    self.rate_limiter.wait_if_needed(estimated_tokens)
                    
                    # Process the prompt
                    result = self.process_single_prompt(prompt_item)
                    if result:
                        dataset.append(result)
                        print(f"Generated example {len(dataset)}/{num_examples}")
                        
                        # Result token usage is tracked within process_single_prompt
                
                # Delay between batches
                if batch_idx < total_batches - 1:
                    batch_delay = min(5 + (batch_idx % 3), 12)
                    print(f"Completed batch {batch_idx+1}/{total_batches}. Waiting {batch_delay} seconds...")
                    time.sleep(batch_delay)
                    
                # Print rate usage stats
                if batch_idx % 5 == 0:
                    usage_stats = self.rate_limiter.get_usage_stats()
                    print(f"Rate limit usage: TPM {usage_stats['tpm_percent']:.1f}%, RPM {usage_stats['rpm_percent']:.1f}%, TPD {usage_stats['tpd_percent']:.1f}%")
            
            print(f"Generated {len(dataset)} examples out of {num_examples} requested")
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Dataset generation failed: {str(e)}")
    
    # =========================================================================
    # OUTPUT AND ANALYSIS METHODS
    # =========================================================================
    
    def save_dataset(self, dataset: List[Dict], output_file: str):
        """
        Save the generated dataset with comprehensive metadata.
        
        Args:
            dataset: List of dataset entries to save
            output_file: Path to save the dataset JSON file
        """
        try:
            # Count finance subtopics
            finance_subtopics = {}
            for item in dataset:
                subtopic = item["metadata"].get("finance_subtopic", "general_finance")
                finance_subtopics[subtopic] = finance_subtopics.get(subtopic, 0) + 1
            
            # Count by complexity type
            complexity_counts = {}
            for item in dataset:
                complexity = item["metadata"].get("complexity_type", "basic")
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
            # Count simplified and truncated items
            simplified_count = sum(1 for item in dataset if item["metadata"].get("question_simplified", False))
            truncated_count = sum(1 for item in dataset if item["metadata"].get("truncated", False))
            
            # Calculate word and token statistics
            question_words = [item["metadata"].get("question_words", 0) for item in dataset]
            response_tokens = [item["metadata"].get("response_tokens", 0) for item in dataset]
            total_tokens = [item["metadata"].get("total_tokens", 0) for item in dataset]
            
            # Add dataset metadata
            dataset_with_meta = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model": self.llm_model,
                    "num_examples": len(dataset),
                    "question_distribution": self.question_categories,
                    "complexity_distribution": complexity_counts,
                    "finance_subtopic_distribution": finance_subtopics,
                    "simplified_questions_count": simplified_count,
                    "truncated_responses_count": truncated_count,
                    "question_words": {
                        "total": sum(question_words),
                        "avg": sum(question_words) / len(question_words) if question_words else 0,
                        "min": min(question_words) if question_words else 0,
                        "max": max(question_words) if question_words else 0
                    },
                    "response_tokens": {
                        "total": sum(response_tokens),
                        "avg": sum(response_tokens) / len(response_tokens) if response_tokens else 0,
                        "min": min(response_tokens) if response_tokens else 0,
                        "max": max(response_tokens) if response_tokens else 0
                    },
                    "total_tokens": {
                        "total": sum(total_tokens),
                        "avg": sum(total_tokens) / len(total_tokens) if total_tokens else 0
                    },
                    "complexity_settings": self.complexity_types,
                    "max_question_words": self.max_question_words,
                    "max_response_tokens": self.max_response_tokens,
                    "domain": "finance",
                    "purpose": "domain-specific vector embedding optimization for financial applications",
                    "data_sources": {
                        "urls": len(self.knowledge_sources.get("urls", [])),
                        "pdfs": len(os.listdir(self.knowledge_sources.get("pdf_dir", ""))) if "pdf_dir" in self.knowledge_sources and os.path.exists(self.knowledge_sources["pdf_dir"]) else 0,
                        "seed_questions": len(self.seed_questions)
                    }
                },
                "data": dataset
            }
            
            # Save to file
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(dataset_with_meta, f, indent=2, ensure_ascii=False)
            print(f"Dataset successfully saved to {output_file}")
            
            # Generate sample for fine-tuning
            fine_tune_sample_file = output_file.replace(".json", "_finetune_sample.jsonl")
            self.generate_fine_tune_sample(dataset[:20], fine_tune_sample_file)
            
            # Print detailed summary statistics
            print("\n=== DATASET SUMMARY STATISTICS ===")
            print(f"Total examples: {len(dataset)}")
            print(f"\nQUESTION STATISTICS:")
            print(f"Average question length: {dataset_with_meta['metadata']['question_words']['avg']:.1f} words")
            print(f"Question length range: {dataset_with_meta['metadata']['question_words']['min']} - {dataset_with_meta['metadata']['question_words']['max']} words")
            print(f"Questions simplified: {simplified_count} ({simplified_count/len(dataset)*100:.1f}%)")
            
            print(f"\nQUESTION COMPLEXITY DISTRIBUTION:")
            for complexity, count in complexity_counts.items():
                print(f"- {complexity}: {count} ({count/len(dataset)*100:.1f}%)")
                
            print(f"\nQUESTION TYPE DISTRIBUTION:")
            for qtype, count in self.question_categories.items():
                print(f"- {qtype}: {count} ({count/len(dataset)*100:.1f}%)")
                
            print(f"\nFINANCIAL SUBTOPIC DISTRIBUTION:")
            for subtopic, count in finance_subtopics.items():
                print(f"- {subtopic}: {count} ({count/len(dataset)*100:.1f}%)")
            
            print(f"\nRESPONSE STATISTICS:")
            print(f"Average response length: {dataset_with_meta['metadata']['response_tokens']['avg']:.1f} tokens")
            print(f"Response length range: {dataset_with_meta['metadata']['response_tokens']['min']} - {dataset_with_meta['metadata']['response_tokens']['max']} tokens")
            print(f"Responses truncated: {truncated_count} ({truncated_count/len(dataset)*100:.1f}%)")
            
            print(f"\nTOKEN USAGE:")
            print(f"Total tokens used: {dataset_with_meta['metadata']['total_tokens']['total']} (est. {dataset_with_meta['metadata']['total_tokens']['total']/1000:.1f}k tokens)")
            
            # Print fine-tuning sample information
            print(f"\nFine-tuning sample created at: {fine_tune_sample_file}")
            
            # Estimate remaining daily capacity
            daily_limit = self.rate_limits["tpd"]
            total_tokens = dataset_with_meta['metadata']['total_tokens']['total']
            remaining = daily_limit - total_tokens
            
            print(f"\nDAILY TOKEN USAGE ESTIMATE:")
            print(f"Used: {total_tokens} tokens ({total_tokens/daily_limit*100:.1f}% of daily limit)")
            print(f"Remaining capacity: ~{remaining} tokens (est. {remaining/total_tokens*len(dataset):.0f} more examples)")
            if total_tokens > daily_limit:
                print("WARNING: Dataset generation has potentially exceeded the daily token limit!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save dataset: {str(e)}")
    
    def generate_fine_tune_sample(self, sample_data: List[Dict], output_file: str):
        """
        Generate a sample file in a format suitable for embedding fine-tuning.
        
        Args:
            sample_data: List of dataset entries to use for the sample
            output_file: Path to save the JSONL sample file
        """
        try:
            # Create example pairs for contrastive learning
            finetune_pairs = []
            
            # Generate positive pairs (same question-answer)
            for i, item in enumerate(sample_data):
                question = item["input"]
                answer = item["output"]
                
                # Create positive pair (question-answer)
                finetune_pairs.append({
                    "text1": question, 
                    "text2": answer, 
                    "label": 1,  # Positive pair
                    "id": f"positive_{i}",
                    "metadata": {
                        "complexity": item["metadata"].get("complexity_type", "basic"),
                        "question_type": item["metadata"].get("question_type", "factual"),
                        "subtopic": item["metadata"].get("finance_subtopic", "general_finance")
                    }
                })
            
            # Generate negative pairs, trying to match complexity and question type
            for i, item in enumerate(sample_data):
                question = item["input"]
                
                # Get complexity and question type
                complexity = item["metadata"].get("complexity_type", "basic")
                question_type = item["metadata"].get("question_type", "factual")
                
                # Find a non-matching item with similar characteristics
                candidates = [
                    j for j, other_item in enumerate(sample_data)
                    if (j != i and 
                        other_item["metadata"].get("complexity_type", "basic") == complexity and
                        other_item["metadata"].get("question_type", "factual") == question_type)
                ]
                
                # If no matching candidates, just select a different item
                if not candidates:
                    candidates = [j for j in range(len(sample_data)) if j != i]
                
                if candidates:
                    negative_idx = random.choice(candidates)
                    negative_answer = sample_data[negative_idx]["output"]
                    
                    finetune_pairs.append({
                        "text1": question, 
                        "text2": negative_answer, 
                        "label": 0,  # Negative pair
                        "id": f"negative_{i}",
                        "metadata": {
                            "complexity": complexity,
                            "question_type": question_type,
                            "subtopic": item["metadata"].get("finance_subtopic", "general_finance")
                        }
                    })
            
            # Write to JSONL format
            with open(output_file, "w", encoding='utf-8') as f:
                for pair in finetune_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    
            print(f"Created {len(finetune_pairs)} contrastive learning pairs for fine-tuning")
            
        except Exception as e:
            print(f"Warning: Failed to generate fine-tuning sample: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it.")

    # Knowledge sources configuration
    knowledge_sources = {
        'urls': [
            "https://www.investopedia.com/terms/c/cryptocurrency.asp",
            "https://www.fool.com/investing/stock-market/market-sectors/financials/cryptocurrency-stocks/",
            "https://www.investors.com/research/magnificent-seven-stocks/"
        ],
        'pdf_dir': '/Users/miteshsingh/Downloads/finance_pdfs'  # Directory containing financial PDFs
    }

    # Load seed questions from prompt.txt if available
    seed_questions = []
    if os.path.exists("prompt.txt"):
        with open("prompt.txt", "r", encoding="utf-8") as f:
            seed_questions = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(seed_questions)} seed questions from prompt.txt")
    else:
        print("prompt.txt not found. Please create it with seed questions or provide them directly.")

    # Set token limits and allow question complexity
    max_response_tokens = 500  # Limit each response to 500 tokens
    max_question_words = 55    # Base limit for simple questions
    allow_complex_questions = True  # Allow more complex multi-part questions
    
    try:
        # Initialize the dataset generator
        print("Initializing financial dataset generator...")
        generator = FinancialDatasetGenerator(
            knowledge_sources=knowledge_sources,
            seed_questions=seed_questions,
            api_key=groq_api_key,
            max_response_tokens=max_response_tokens,
            max_question_words=max_question_words,
            allow_complex_questions=allow_complex_questions
        )

        # Generate dataset (adjust number as needed)
        print("Generating synthetic dataset...")
        dataset = generator.generate_dataset(num_examples=1000)
        
        # Save and analyze dataset
        print("Saving dataset...")
        generator.save_dataset(dataset, "financial_domain_dataset.json")
        
        print("Dataset generation complete!")
        
    except Exception as e:
        print(f"Error in dataset generation: {str(e)}")