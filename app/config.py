import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please get one from Google AI Studio.")

if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY not found in .env file. Please get one from exa.ai.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file. Please get one from pinecone.")

if not PINECONE_ENV:
    raise ValueError("PINECONE_ENV not found in .env file.")