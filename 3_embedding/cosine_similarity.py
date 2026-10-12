import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)

document = [
    "I want to learn Python programming.",
    "I want to study Python.",
    "The weather is sunny today.",
]

query = "Tell me about AI and Python."


document_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=document,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT"
    ),
)

document_vectors = [
    embedding.values 
    for  embedding in document_response.embeddings
]



query_response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    ),
)

query_vector = query_response.embeddings[0].values

scores = cosine_similarity(
    [query_vector],
    document_vectors
)[0]
# print(scores)


ranked_results = sorted(
    zip(document, scores),
    key=lambda item: item[1],
    reverse=True,
)

# print(ranked_results)

# 10. Display ranked results
for document, score in ranked_results:
    print(f"Score: {score:.4f} | {document}")



