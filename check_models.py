import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("--- Available Models ---")

# Iterate through the list of available models
for m in genai.list_models():
    # Check if the model supports the 'generateContent' method
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")

print("------------------------")