

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os



# === Setup ===
# Display settings
pd.set_option('display.max_columns', 100)
sns.set(style="whitegrid")

# === Load Data ===
# Load dataset from local path
DATA_PATH = "../data/complaints.csv"
df = pd.read_csv(DATA_PATH)

# Preview
df.head()

# === Initial Exploration ===
# Dataset dimensions
print(f"Total records: {len(df)}")
print(df.info())

# Check missing values
missing_summary = df.isnull().sum()
print("\nMissing Values Summary:\n", missing_summary)

# Complaints with and without narratives
has_narrative = df["Consumer complaint narrative"].notnull().sum()
no_narrative = df["Consumer complaint narrative"].isnull().sum()
print(f"\nComplaints with narrative: {has_narrative}")
print(f"Complaints without narrative: {no_narrative}")

# === Product Distribution ===
# Top-level product distribution
product_counts = df["Product"].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=product_counts.index, y=product_counts.values)
plt.xticks(rotation=45)
plt.title("Distribution of Complaints Across Products")
plt.ylabel("Number of Complaints")
plt.tight_layout()
plt.show()

# === Narrative Length Analysis ===
# Add word count column
df["narrative_length"] = df["Consumer complaint narrative"].fillna("").apply(lambda x: len(x.split()))

# Histogram of narrative lengths
plt.figure(figsize=(10, 4))
sns.histplot(df["narrative_length"], bins=50, kde=True)
plt.title("Distribution of Complaint Narrative Lengths (in words)")
plt.xlabel("Word Count")
plt.ylabel("Number of Complaints")
plt.show()

# === Filter & Clean ===
# Define product categories of interest
target_products = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later",
    "Savings account",
    "Money transfer, virtual currency, or money service"
]

# Filter to target products with non-empty narratives
filtered_df = df[
    df["Product"].isin(target_products) &
    df["Consumer complaint narrative"].notnull()
].copy()

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove special characters
    text = re.sub(r"\b(i am writing to file a complaint.*?)\b", " ", text)  # Remove boilerplate if it appears
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    return text.strip()

filtered_df["cleaned_narrative"] = filtered_df["Consumer complaint narrative"].apply(clean_text)

# === Save Results ===
os.makedirs("../data", exist_ok=True)
filtered_df.to_csv("../data/filtered_complaints.csv", index=False)

print(f"\nFiltered and cleaned dataset saved. Rows: {len(filtered_df)}")
