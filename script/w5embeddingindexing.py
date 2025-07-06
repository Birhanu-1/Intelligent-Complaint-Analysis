

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle

# === Load Filtered Data ===
df = pd.read_csv("../data/filtered_complaints.csv")

# === Chunking Strategy ===
chunk_size = 500
chunk_overlap = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)

# Prepare chunks and metadata
texts = []
metadatas = []

for _, row in df.iterrows():
    chunks = text_splitter.split_text(row["cleaned_narrative"])
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({
            "complaint_id": row["Complaint ID"],
            "product": row["Product"]
        })

print(f"Total chunks created: {len(texts)}")

# === Embedding Model ===
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# === FAISS Indexing ===
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save FAISS index and metadata
os.makedirs("..data/vector_store", exist_ok=True)
faiss.write_index(index, "..data/vector_store/faiss_index.index")

# Save metadata mapping
with open("..data/vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)

print("Vector store and metadata saved to ../vector_store/")
