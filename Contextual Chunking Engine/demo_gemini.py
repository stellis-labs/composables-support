import os
import re
import gradio as gr
import pandas as pd
import numpy as np
import json
import time
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from PyPDF2 import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup
from configparser import ConfigParser
import traceback
import logging
from datetime import datetime
import google.generativeai as genai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ContextualChunker")


class ChunkerConfig:
    """Configuration manager for the chunking engine"""

    def __init__(self, config_path=None):
        self.config = ConfigParser()

        # Default configuration
        self.defaults = {
            "api": {
                "provider": "gemini",
                "model": "gemini-flash",
                "api_key": os.environ.get("GOOGLE_API_KEY", ""),
                "temperature": 0.1,
                "timeout": 30
            },
            "chunking": {
                "use_llm": "true",
                "technical_chunk_size": 1500,
                "technical_overlap": 100,
                "conversational_chunk_size": 800,
                "conversational_overlap": 30,
                "mixed_chunk_size": 1000,
                "mixed_overlap": 50,
                "default_chunk_size": 1000,
                "default_overlap": 50,
                "min_chunk_size": 150
            },
            "processing": {
                "max_retries": 3,
                "retry_delay": 2,
                "batch_size": 10000
            }
        }

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.config.read(config_path)
        else:
            # Initialize with defaults
            for section, options in self.defaults.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                for key, value in options.items():
                    self.config.set(section, key, str(value))

    def get(self, section, option, fallback=None):
        """Get a configuration value"""
        return self.config.get(section, option, fallback=fallback)

    def getboolean(self, section, option, fallback=None):
        """Get a boolean configuration value"""
        return self.config.getboolean(section, option, fallback=fallback)

    def getint(self, section, option, fallback=None):
        """Get an integer configuration value"""
        return self.config.getint(section, option, fallback=fallback)

    def getfloat(self, section, option, fallback=None):
        """Get a float configuration value"""
        return self.config.getfloat(section, option, fallback=fallback)

    def set(self, section, option, value):
        """Set a configuration value"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))

    def save(self, config_path):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            self.config.write(f)

    def update_api_key(self, api_key):
        """Update the API key"""
        self.set("api", "api_key", api_key)
        # Also update environment variable
        os.environ["GOOGLE_API_KEY"] = api_key


class LLMClient:
    """Client for interacting with LLM APIs"""

    def __init__(self, config):
        self.config = config
        self.provider = config.get("api", "provider", fallback="gemini")
        self.model = config.get("api", "model", fallback="gemini-flash")
        self.api_key = config.get("api", "api_key", fallback="")
        self.temperature = config.getfloat("api", "temperature", fallback=0.1)
        self.timeout = config.getint("api", "timeout", fallback=30)
        self.max_retries = config.getint("processing", "max_retries", fallback=3)
        self.retry_delay = config.getint("processing", "retry_delay", fallback=2)
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if self.provider == "gemini":
            try:
                # Configure the Gemini API
                genai.configure(api_key=self.api_key)
                self.client = genai
                logger.info(f"Initialized Google AI client with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI client: {str(e)}")
                self.client = None
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            self.client = None

    def is_available(self):
        """Check if the LLM client is available"""
        return self.client is not None and self.api_key

    def query(self, prompt, max_tokens=1000):
        """Send a query to the LLM with retry logic"""
        if not self.is_available():
            logger.error("LLM client not available")
            return None

        for attempt in range(self.max_retries):
            try:
                # Use the Gemini model for generation
                model = self.client.GenerativeModel(model_name=self.model,
                                                    generation_config={"temperature": self.temperature,
                                                                       "max_output_tokens": max_tokens})

                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"LLM query failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All retry attempts failed.")
                    return None

    def parse_json_response(self, response):
        """Parse JSON response from LLM"""
        if not response:
            return None

        try:
            # Try to extract JSON if it's wrapped in backticks or markdown
            if "```json" in response and "```" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response[:100]}...")
            return None


class ContentAnalyzer:
    """Analyzes document content to determine type and structure"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.technical_terms = [
            "algorithm", "analysis", "function", "equation", "parameter",
            "variable", "coefficient", "theorem", "methodology", "implementation",
            "protocol", "specification", "configuration", "interface", "module",
            "component", "architecture", "framework", "infrastructure", "deployment"
        ]
        self.conversational_terms = [
            "think", "feel", "believe", "opinion", "chat", "talk", "discuss",
            "conversation", "dialogue", "speaking", "listening", "agree", "disagree",
            "thanks", "please", "welcome", "hello", "goodbye", "hey", "hi"
        ]

    def analyze_with_llm(self, text):
        """Analyze content using LLM"""
        prompt = f"""
        Analyze the following document content and identify its structural properties and semantic characteristics.

        Provide your analysis in JSON format with the following fields:
        1. "content_type": The general type of content (one of: "technical", "conversational", "mixed", "narrative", "informational")
        2. "structure_score": A score from 1-10 indicating how structured the content is (1=very unstructured, 10=highly structured)
        3. "key_sections": An array of the main semantic sections identified (up to 5)
        4. "technical_ratio": An estimate (0.0-1.0) of how technical the content is
        5. "complexity_score": A score from 1-10 indicating how complex the content is
        6. "suggested_chunk_size": Suggested optimal chunk size in characters for this content
        7. "suggested_overlap": Suggested optimal overlap size in characters for this content

        Here is the content to analyze:
        {text[:5000]}

        JSON response:
        """

        response = self.llm_client.query(prompt)
        return self.llm_client.parse_json_response(response) or self._fallback_analysis(text)

    def analyze_with_heuristics(self, text):
        """Analyze content using heuristics"""
        # Technical indicators
        technical_indicators = 0

        # Count acronyms (uppercase words of 2-5 letters)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', text)
        technical_indicators += len(acronyms)

        # Count decimal numbers
        decimal_numbers = re.findall(r'\b\d+\.\d+\b', text)
        technical_indicators += len(decimal_numbers)

        # Count figure/table references
        figure_refs = re.findall(r'(figure|fig\.|table|tab\.)\s*\d+', text, re.IGNORECASE)
        technical_indicators += len(figure_refs)

        # Count technical terms
        tech_term_count = sum(
            1 for term in self.technical_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE))
        technical_indicators += tech_term_count

        # Conversational indicators
        conversational_indicators = 0

        # Count personal pronouns
        pronouns = re.findall(r'\b(I|we|you|he|she|they|me|us|him|her|them)\b', text, re.IGNORECASE)
        conversational_indicators += len(pronouns)

        # Count questions
        questions = re.findall(r'\?', text)
        conversational_indicators += len(questions)

        # Count exclamations
        exclamations = re.findall(r'!', text)
        conversational_indicators += len(exclamations)

        # Count conversational terms
        conv_term_count = sum(
            1 for term in self.conversational_terms if re.search(r'\b' + term + r'\b', text, re.IGNORECASE))
        conversational_indicators += conv_term_count

        # Calculate ratios based on text length (per 1000 chars)
        text_length = max(1, len(text) / 1000)  # Per 1000 characters
        technical_ratio = technical_indicators / text_length
        conversational_ratio = conversational_indicators / text_length

        # Determine content type
        if technical_ratio > 5 and technical_ratio > conversational_ratio * 2:
            content_type = "technical"
            structure_score = min(8, 5 + technical_ratio / 5)
            suggested_chunk_size = 1500
            suggested_overlap = 100
        elif conversational_ratio > 5 and conversational_ratio > technical_ratio * 2:
            content_type = "conversational"
            structure_score = max(3, 5 - conversational_ratio / 5)
            suggested_chunk_size = 800
            suggested_overlap = 30
        else:
            content_type = "mixed"
            structure_score = 5
            suggested_chunk_size = 1000
            suggested_overlap = 50

        # Create analysis result
        return {
            "content_type": content_type,
            "structure_score": round(structure_score, 1),
            "key_sections": [],
            "technical_ratio": round(technical_ratio / 10, 2),  # Scale to 0-1
            "complexity_score": round((technical_ratio + structure_score) / 3, 1),
            "suggested_chunk_size": suggested_chunk_size,
            "suggested_overlap": suggested_overlap
        }

    def _fallback_analysis(self, text):
        """Provide fallback analysis when LLM fails"""
        return self.analyze_with_heuristics(text)

    def analyze(self, text, use_llm=True):
        """Analyze content with LLM if available, fallback to heuristics otherwise"""
        if use_llm and self.llm_client.is_available():
            llm_analysis = self.analyze_with_llm(text)
            if llm_analysis:
                return llm_analysis

        # Fallback to heuristics
        return self.analyze_with_heuristics(text)


class SmartSentenceSplitter:
    """Advanced sentence splitter that handles edge cases better than NLTK"""

    def __init__(self):
        # Common abbreviations to avoid splitting after
        self.abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "Ltd.", "Co.",
            "Jr.", "Sr.", "St.", "Rd.", "Ave.", "Blvd.", "Apt.", "Dept.",
            "Fig.", "et al.", "i.e.", "e.g."
        ]

    def split(self, text):
        """Split text into sentences without relying on complex regex lookbehinds"""
        # First, tokenize by potential sentence endings
        raw_splits = re.split(r'([.!?])\s+', text)

        # Reconstruct the splits with their punctuation
        potential_sentences = []
        i = 0
        while i < len(raw_splits) - 1:
            if i + 1 < len(raw_splits):
                potential_sentences.append(raw_splits[i] + raw_splits[i + 1])
                i += 2
            else:
                potential_sentences.append(raw_splits[i])
                i += 1

        # If there's a leftover piece, add it
        if i < len(raw_splits):
            potential_sentences.append(raw_splits[i])

        # Process potential sentences to handle abbreviations
        sentences = []
        current = ""

        for s in potential_sentences:
            # Check if current ends with an abbreviation
            is_abbreviation = False
            for abbr in self.abbreviations:
                if current.strip().endswith(abbr):
                    is_abbreviation = True
                    break

            # If the current text ends with an abbreviation,
            # or the next sentence doesn't start with a capital letter,
            # it's likely not a sentence boundary
            if is_abbreviation or not (s.strip() and s.strip()[0].isupper()):
                current += " " + s
            else:
                if current:
                    sentences.append(current.strip())
                current = s

        # Don't forget the last part
        if current.strip():
            sentences.append(current.strip())

        return sentences


class SemanticBreakFinder:
    """Finds semantic break points in text"""

    def __init__(self, llm_client, config, content_analyzer):
        self.llm_client = llm_client
        self.config = config
        self.content_analyzer = content_analyzer
        self.sentence_splitter = SmartSentenceSplitter()

    def find_breaks_with_llm(self, text, content_analysis, target_chunk_size):
        """Find semantic break points using LLM"""
        prompt = f"""
        I need to divide the following text into semantically meaningful chunks of approximately {target_chunk_size} characters each.

        The content has been analyzed as: {content_analysis['content_type']} content with a complexity score of {content_analysis['complexity_score']}/10.

        Please identify the best places to split this text to preserve context and meaning.
        For each suggested split point, give me:
        1. The character index where the split should occur
        2. A score from 0-1 indicating how good this split point is (1 = perfect natural break, 0 = bad place to split)
        3. A brief reason why this is a good split point

        Return your answer as a JSON array of objects, each with "index", "score", and "reason" fields.

        Here's the text:
        {text[:12000]}

        JSON response:
        """

        response = self.llm_client.query(prompt)
        result = self.llm_client.parse_json_response(response)

        if result:
            # If result is a dictionary with a "splits" key, extract that
            if isinstance(result, dict) and "splits" in result:
                return result["splits"]
            # If result is already a list, return it
            elif isinstance(result, list):
                return result

        # If we couldn't get results from LLM, return empty list
        return None

    def find_breaks_with_heuristics(self, text, content_analysis, target_chunk_size):
        """Find semantic break points using heuristics"""
        breaks = []

        # Split text into sentences
        sentences = self.sentence_splitter.split(text)

        # Find potential paragraph boundaries (double newlines)
        paragraph_boundaries = [m.start() for m in re.finditer(r'\n\s*\n', text)]

        # Find potential section headers
        section_patterns = [
            r'#+\s+[A-Z]',  # Markdown headers
            r'^[A-Z][A-Za-z\s]+:',  # Title followed by colon
            r'^[IVX]+\.\s+[A-Z]',  # Roman numeral sections
            r'^[0-9]+\.\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+$'  # ALL CAPS headers
        ]

        section_boundaries = []
        for pattern in section_patterns:
            section_boundaries.extend([m.start() for m in re.finditer(pattern, text, re.MULTILINE)])

        # Combine and sort all potential break points
        all_boundaries = []

        # Add sentence boundaries with moderate scores
        char_index = 0
        for sentence in sentences:
            char_index += len(sentence)
            all_boundaries.append({
                "index": char_index,
                "score": 0.3,
                "reason": "Sentence boundary"
            })

        # Add paragraph boundaries with higher scores
        for idx in paragraph_boundaries:
            all_boundaries.append({
                "index": idx,
                "score": 0.7,
                "reason": "Paragraph boundary"
            })

        # Add section boundaries with highest scores
        for idx in section_boundaries:
            all_boundaries.append({
                "index": idx,
                "score": 0.9,
                "reason": "Section header"
            })

        # Sort by index
        all_boundaries.sort(key=lambda x: x["index"])

        # Filter boundaries to create chunks of approximately target_chunk_size
        current_index = 0
        while current_index < len(text):
            target_index = current_index + target_chunk_size

            # Find boundaries within +/- 30% of target size
            candidates = [b for b in all_boundaries if b["index"] > current_index and
                          abs(b["index"] - target_index) < 0.3 * target_chunk_size]

            if candidates:
                # Pick the best candidate based on score
                best_candidate = max(candidates, key=lambda x: x["score"])
                breaks.append(best_candidate)
                current_index = best_candidate["index"]
            else:
                # If no good candidates, just use target index
                breaks.append({
                    "index": min(target_index, len(text)),
                    "score": 0.1,
                    "reason": "Forced break due to size constraints"
                })
                current_index = target_index

        return breaks

    def find_breaks(self, text, content_analysis, use_llm=True):
        """Find semantic break points in text"""
        # Determine target chunk size based on content type
        content_type = content_analysis.get("content_type", "mixed")
        if content_type == "technical":
            target_chunk_size = self.config.getint("chunking", "technical_chunk_size",
                                                   fallback=content_analysis.get("suggested_chunk_size", 1500))
        elif content_type == "conversational":
            target_chunk_size = self.config.getint("chunking", "conversational_chunk_size",
                                                   fallback=content_analysis.get("suggested_chunk_size", 800))
        else:  # mixed or other
            target_chunk_size = self.config.getint("chunking", "mixed_chunk_size",
                                                   fallback=content_analysis.get("suggested_chunk_size", 1000))

        # Try to find breaks with LLM if enabled
        if use_llm and self.llm_client.is_available():
            llm_breaks = self.find_breaks_with_llm(text, content_analysis, target_chunk_size)
            if llm_breaks:
                return llm_breaks

        # Fallback to heuristics
        return self.find_breaks_with_heuristics(text, content_analysis, target_chunk_size)


class ChunkMetadata:
    """Metadata for a chunk"""

    def __init__(self, index, start, end, text, score=0, reason="", prev_id=None, next_id=None):
        self.id = f"chunk-{index}"
        self.index = index
        self.start = start
        self.end = end
        self.length = end - start
        self.text = text
        self.score = score
        self.reason = reason
        self.prev_id = prev_id
        self.next_id = next_id
        self.content_summary = ""
        self.word_count = len(text.split())
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "score": self.score,
            "reason": self.reason,
            "prev_id": self.prev_id,
            "next_id": self.next_id,
            "content_summary": self.content_summary,
            "word_count": self.word_count,
            "created_at": self.created_at,
            "preview": self.text[:100] + "..." if len(self.text) > 100 else self.text
        }


class ContextualChunker:
    """Main chunking engine that preserves context"""

    def __init__(self, config_path=None):
        self.config = ChunkerConfig(config_path)
        self.llm_client = LLMClient(self.config)
        self.content_analyzer = ContentAnalyzer(self.llm_client)
        self.break_finder = SemanticBreakFinder(self.llm_client, self.config, self.content_analyzer)

        self.document_text = ""
        self.chunks = []
        self.chunk_metadata = []
        self.content_analysis = {}
        self.performance_metrics = {}

        self.use_llm = self.config.getboolean("chunking", "use_llm", fallback=True)

    def extract_text(self, file_path):
        """Extract text from various file formats"""
        start_time = time.time()
        temp_file_path = None

        try:
            # Handle Gradio file upload (file_path will be the path string in this case)
            if isinstance(file_path, str):
                file_path_str = file_path
            # Handle file-like object with read method
            elif hasattr(file_path, 'read') and callable(file_path.read):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path.name)[1]) as temp_file:
                    temp_file.write(file_path.read())
                    temp_file_path = temp_file.name
                file_path_str = temp_file_path
            # Handle Gradio file component upload which provides a file object with name attribute
            elif hasattr(file_path, 'name'):
                file_path_str = file_path.name
            else:
                raise ValueError(f"Unsupported file object type: {type(file_path)}")

            try:
                if file_path_str.lower().endswith('.pdf'):
                    text = self._extract_from_pdf(file_path_str)
                elif file_path_str.lower().endswith('.docx'):
                    text = self._extract_from_docx(file_path_str)
                elif file_path_str.lower().endswith('.txt'):
                    text = self._extract_from_txt(file_path_str)
                elif file_path_str.lower().endswith('.md'):
                    text = self._extract_from_markdown(file_path_str)
                else:
                    raise ValueError(f"Unsupported file format: {file_path_str}")

                # Clean up the text
                text = self._clean_text(text)

                # Record performance metrics
                self.performance_metrics["extraction_time"] = time.time() - start_time
                self.performance_metrics["document_length"] = len(text)

                # Store document text
                self.document_text = text

                return text

            except Exception as e:
                logger.error(f"Error extracting text: {str(e)}")
                logger.error(traceback.format_exc())
                raise
        finally:
            # Delete temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")

    def _extract_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _extract_from_docx(self, file_path):
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise

    def _extract_from_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            logger.error("Failed to decode text with multiple encodings")
            raise

    def _extract_from_markdown(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
                html = markdown.markdown(md_text)
                soup = BeautifulSoup(html, features='html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error extracting text from Markdown: {str(e)}")
            raise

    def _clean_text(self, text):
        """Clean extracted text"""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        # Remove non-printable characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        return text

    def analyze_content(self, text):
        """Analyze document content"""
        start_time = time.time()

        # Analyze content
        self.content_analysis = self.content_analyzer.analyze(text, use_llm=self.use_llm)

        # Record performance metrics
        self.performance_metrics["analysis_time"] = time.time() - start_time

        return self.content_analysis

    def create_chunks(self, text, use_llm=None):
        """Create semantically meaningful chunks"""
        if use_llm is None:
            use_llm = self.use_llm

        start_time = time.time()

        # Reset chunks
        self.chunks = []
        self.chunk_metadata = []

        # If no content analysis yet, analyze content
        if not self.content_analysis:
            self.analyze_content(text)

        # If text is shorter than min chunk size, return as single chunk
        min_chunk_size = self.config.getint("chunking", "min_chunk_size", fallback=150)
        if len(text) < min_chunk_size:
            self.chunks = [text]
            self.chunk_metadata = [ChunkMetadata(0, 0, len(text), text, 1.0, "Complete document")]
            return self.chunks

        # Find semantic break points
        break_points = self.break_finder.find_breaks(text, self.content_analysis, use_llm)

        # Get appropriate overlap based on content type
        content_type = self.content_analysis.get("content_type", "mixed")
        if content_type == "technical":
            overlap = self.config.getint("chunking", "technical_overlap", fallback=100)
        elif content_type == "conversational":
            overlap = self.config.getint("chunking", "conversational_overlap", fallback=30)
        else:  # mixed or other
            overlap = self.config.getint("chunking", "mixed_overlap", fallback=50)

        # Create chunks based on break points
        start_idx = 0
        for i, break_point in enumerate(break_points):
            end_idx = break_point["index"]
            score = break_point.get("score", 0.5)
            reason = break_point.get("reason", "Unknown")

            # Calculate chunk text
            chunk_text = text[start_idx:end_idx]

            # Create chunk metadata
            prev_id = f"chunk-{i - 1}" if i > 0 else None
            next_id = f"chunk-{i + 1}" if i < len(break_points) - 1 else None
            chunk_metadata = ChunkMetadata(i, start_idx, end_idx, chunk_text, score, reason, prev_id, next_id)

            # Add to chunks
            self.chunks.append(chunk_text)
            self.chunk_metadata.append(chunk_metadata)

            # Update start index for next chunk with overlap
            start_idx = max(0, end_idx - overlap)

        # Add final chunk if needed
        if start_idx < len(text):
            chunk_text = text[start_idx:]
            i = len(self.chunks)
            prev_id = f"chunk-{i - 1}" if i > 0 else None
            chunk_metadata = ChunkMetadata(i, start_idx, len(text), chunk_text, 1.0, "End of document", prev_id, None)

            self.chunks.append(chunk_text)
            self.chunk_metadata.append(chunk_metadata)

        # Update next_id for the last regular chunk
        if len(self.chunk_metadata) >= 2:
            self.chunk_metadata[-2].next_id = self.chunk_metadata[-1].id

        # Record performance metrics
        self.performance_metrics["chunking_time"] = time.time() - start_time
        self.performance_metrics["chunk_count"] = len(self.chunks)
        self.performance_metrics["avg_chunk_size"] = sum(len(chunk) for chunk in self.chunks) / max(1, len(self.chunks))

        return self.chunks

    def get_chunk_metadata(self):
        """Get detailed information about each chunk"""
        return [metadata.to_dict() for metadata in self.chunk_metadata]

    def get_performance_metrics(self):
        """Get performance metrics"""
        return self.performance_metrics

    def toggle_llm(self, use_llm):
        """Toggle LLM usage"""
        self.use_llm = use_llm
        self.config.set("chunking", "use_llm", str(use_llm).lower())

    def update_api_key(self, api_key):
        """Update API key"""
        self.config.update_api_key(api_key)
        self.llm_client = LLMClient(self.config)  # Reinitialize client with new key


def process_document(file, api_key=None, use_llm=True, chunk_params=None):
    """Process a document with the contextual chunker"""
    try:
        # Initialize chunker
        chunker = ContextualChunker()

        # Update API key if provided
        if api_key:
            chunker.update_api_key(api_key)

        # Toggle LLM usage
        chunker.toggle_llm(use_llm)

        # Update chunk parameters if provided
        if chunk_params and isinstance(chunk_params, dict):
            for key, value in chunk_params.items():
                if isinstance(value, (int, float)):
                    chunker.config.set("chunking", key, str(value))

        # Extract text from document
        text = chunker.extract_text(file)

        # Analyze content
        content_analysis = chunker.analyze_content(text)

        # Create chunks
        chunks = chunker.create_chunks(text)

        # Get chunk details
        chunk_details = chunker.get_chunk_metadata()

        # Get performance metrics
        metrics = chunker.get_performance_metrics()

        # Format status message
        status = (
            f"Created {len(chunks)} contextual chunks\n"
            f"Content type: {content_analysis.get('content_type', 'unknown')}\n"
            f"Structure score: {content_analysis.get('structure_score', 0)}/10\n"
            f"Avg chunk size: {metrics.get('avg_chunk_size', 0):.0f} chars\n"
            f"Processing time: {sum(v for k, v in metrics.items() if k.endswith('_time')):.2f} seconds"
        )

        return text, chunks, chunk_details, content_analysis, metrics, status

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None, f"Error: {str(e)}"


# Gradio UI
with gr.Blocks(title="Contextual Chunking Engine") as demo:
    gr.Markdown("# Contextual Chunking Engine Demo")
    gr.Markdown("Upload a document to analyze and create dynamic contextual chunks that preserve meaning")

    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(label="Upload Document (PDF, DOCX, TXT, MD)")

        with gr.Column(scale=2):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Google AI API Key",
                    placeholder="Enter your Google AI API key...",
                    type="password"
                )

            with gr.Row():
                use_llm_checkbox = gr.Checkbox(
                    label="Use LLM for enhanced chunking",
                    value=True
                )

    with gr.Row():
        process_btn = gr.Button("Process Document", variant="primary")

    with gr.Row():
        status = gr.Textbox(label="Status")

    with gr.Tabs():
        with gr.TabItem("Created Chunks"):
            chunks_output = gr.Dataframe(
                headers=["Chunk #", "Length", "Preview"],
                label="Generated Chunks"
            )

        with gr.TabItem("Performance Metrics"):
            metrics_output = gr.JSON(label="Performance Metrics")

        with gr.TabItem("Original Text"):
            text_output = gr.Textbox(label="Extracted Text", lines=20)

        with gr.TabItem("Content Analysis JSON"):
            content_analysis_output = gr.JSON(label="Content Analysis Details")

        with gr.TabItem("Chunk Metadata JSON"):
            chunk_metadata_output = gr.JSON(label="Complete Chunk Metadata")

    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            with gr.Column():
                technical_chunk_size = gr.Slider(
                    minimum=500, maximum=3000, value=1500, step=100,
                    label="Technical Content Chunk Size"
                )
                technical_overlap = gr.Slider(
                    minimum=50, maximum=300, value=100, step=10,
                    label="Technical Content Overlap"
                )

            with gr.Column():
                conversational_chunk_size = gr.Slider(
                    minimum=300, maximum=1500, value=800, step=100,
                    label="Conversational Content Chunk Size"
                )
                conversational_overlap = gr.Slider(
                    minimum=20, maximum=150, value=30, step=10,
                    label="Conversational Content Overlap"
                )

            with gr.Column():
                mixed_chunk_size = gr.Slider(
                    minimum=400, maximum=2000, value=1000, step=100,
                    label="Mixed Content Chunk Size"
                )
                mixed_overlap = gr.Slider(
                    minimum=30, maximum=200, value=50, step=10,
                    label="Mixed Content Overlap"
                )


    # Function to process and format chunk data for display
    def format_chunks_for_display(chunks, chunk_details):
        if not chunks or not chunk_details:
            return []

        formatted_chunks = []
        for i, (chunk, metadata) in enumerate(zip(chunks, chunk_details)):
            if isinstance(metadata, ChunkMetadata):
                metadata = metadata.to_dict()

            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            formatted_chunks.append([
                f"Chunk {i + 1}",
                len(chunk),
                preview
            ])

        return formatted_chunks


    # Process button click event
    def on_process_click(file, api_key, use_llm, tech_size, tech_overlap,
                         conv_size, conv_overlap, mixed_size, mixed_overlap):
        # Create parameters dictionary with properly converted values
        chunk_params = {
            "technical_chunk_size": int(tech_size),
            "technical_overlap": int(tech_overlap),
            "conversational_chunk_size": int(conv_size),
            "conversational_overlap": int(conv_overlap),
            "mixed_chunk_size": int(mixed_size),
            "mixed_overlap": int(mixed_overlap)
        }

        # Process the document
        text, chunks, chunk_details, content_analysis, metrics, status_msg = process_document(
            file, api_key, use_llm, chunk_params
        )

        # Format chunks for display
        formatted_chunks = []
        if chunks and chunk_details:
            for i, (chunk, details) in enumerate(zip(chunks, chunk_details)):
                preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                formatted_chunks.append([
                    f"Chunk {i + 1}",
                    len(chunk),
                    preview
                ])

        return text, formatted_chunks, metrics, chunk_details, content_analysis, status_msg


    # Process button click event
    process_btn.click(
        on_process_click,
        inputs=[
            file_input,
            api_key_input,
            use_llm_checkbox,
            technical_chunk_size,
            technical_overlap,
            conversational_chunk_size,
            conversational_overlap,
            mixed_chunk_size,
            mixed_overlap
        ],
        outputs=[text_output, chunks_output, metrics_output, chunk_metadata_output, content_analysis_output, status]
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()