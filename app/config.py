import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please get one from Google AI Studio.")

if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY not found in .env file. Please get one from exa.ai.")