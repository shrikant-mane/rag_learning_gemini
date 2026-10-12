import os
from dotenv import load_dotenv
import numpy as np
from google import genai

load_dotenv()

# Read the API key
api_key = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

sentences = [
    "I want to learn Python programming.",
    "I want to study Python.",
    "The weather is sunny today.",
]

# Generate embeddings
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sentences,
)

# Extract vectors into a NumPy array
embeddings = np.array([
    embedding.values
    for embedding in result.embeddings
])

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)

for sentence, embedding in zip(sentences, embeddings):
    print("\nText:", sentence)
    print("Vector dimensions:", len(embedding))
    print("First five values:", embedding[:5])
