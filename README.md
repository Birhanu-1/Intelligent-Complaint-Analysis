# Intelligent-Complaint-Analysis
# Setup & Execution Guide
This project builds a Retrieval-Augmented Generation (RAG) pipeline to analyze financial product complaints. Tasks 1 and 2 cover data cleaning, chunking, embedding, and indexing for semantic search.
# Task 1: Exploratory Data Analysis & Preprocessing
#  Objective:
Prepare the CFPB complaints dataset for semantic search by filtering, cleaning, and structuring the narratives.
Input:
Place your raw CFPB CSV (e.g., Consumer_Complaints.csv) into the data/ directory.
 Steps:
Run the notebook:
,,bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
,,

This will:

- Load and analyze the raw dataset

- Visualize narrative lengths and product distributions

- Filter to 5 core product categories:

- Credit card

- Personal loan

- Buy Now, Pay Later (BNPL)

- Savings account

- Money transfers

- Clean and normalize the complaint narratives
,,bash

data/filtered_complaints.csv
,,

# Task 2: Text Chunking, Embedding, and FAISS Indexing
# Objective:
Convert cleaned complaint narratives into vector embeddings and index them for fast semantic search.
Requirements:
Install dependencies:

,, bash

pip install pandas matplotlib seaborn langchain faiss-cpu sentence-transformers
,, 
If you're using a GPU and want FAISS GPU support:
,,bash 
pip install faiss-gpu

,,
⚙️ Steps:
Run the script:

,, bash
python w5_embedding_indexing.py

,, 
This will:

Split each narrative into chunks (chunk_size=500, chunk_overlap=100)

Generate embeddings using: sentence-transformers/all-MiniLM-L6-v2

Index the embeddings in a FAISS vector store

Save the index and metadata in:

,,bash

vector_store/faiss_index.index
vector_store/metadata.pkl

,,




