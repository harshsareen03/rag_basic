import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load document
# -----------------------------

with open("data/company_policy.txt", "r") as file:
    text = file.read()


# -----------------------------
# 2. Chunking
# -----------------------------

def create_chunks(text, chunk_size=200):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")


# -----------------------------
# 4. Create FAISS index
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# 5. Save FAISS index
# -----------------------------

faiss.write_index(
    index,
    "company_policy.index"
)


# -----------------------------
# 6. Save chunks
# -----------------------------

with open("chunks.json", "w") as file:
    json.dump(chunks, file, indent=4)


print("Index created successfully!")
print("Vectors stored:", index.ntotal)