import os
import json
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ==================================
# Load environment variables
# ==================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )


# ==================================
# Gemini client
# ==================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==================================
# Embedding model
# ==================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================
# Load FAISS index
# ==================================

index = faiss.read_index(
    "company_policy.index"
)


# ==================================
# Load chunks
# ==================================

with open("chunks.json", "r") as file:
    chunks = json.load(file)


# ==================================
# Retrieval
# ==================================

def retrieve(query, k=3):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results


# ==================================
# Create prompt
# ==================================

def create_prompt(question, retrieved_chunks):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a company policy assistant.

Answer the question using ONLY the
information provided in the context.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt


# ==================================
# Gemini
# ==================================

def generate_answer(question):

    retrieved_chunks = retrieve(
        question,
        k=3
    )

    prompt = create_prompt(
        question,
        retrieved_chunks
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text, retrieved_chunks


# ==================================
# Test
# ==================================

question = "How many paid leaves do employees get?"

answer, retrieved_chunks = generate_answer(
    question
)

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

for chunk in retrieved_chunks:
    print("\n---")
    print(chunk)


print("\n==============================")
print("GEMINI ANSWER")
print("==============================")

print(answer)