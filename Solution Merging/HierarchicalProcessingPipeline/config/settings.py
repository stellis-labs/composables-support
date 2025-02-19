import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Loading API Key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initializing Groq client
client = Groq(api_key=GROQ_API_KEY)