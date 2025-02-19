import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MODEL_NAME = "llama3-70b-8192"
    OUTPUT_FILE = "synthetic_dataset.json"
    NUM_ENTRIES = 5000
    PROMPT_TEMPLATE = """Generate a realistic but fictional user profile with these fields:
    - Name: {name}
    - Age: {age}
    - Email: {email}
    - City: {city}
    - Job Title: {job_title}
    - Biography: {bio}
    
    Format the response as JSON only, without any additional text or explanations.
    Use realistic values and maintain diversity in the data."""