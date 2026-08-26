import json
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

EMBEDDING_MODEL = "gemini-embedding-001"

QUERY = "Was James Carter ultimately allowed to keep working for Vertex Metrics?"

TOP_K = 5

"""Directory settings"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "experiments" / "first_embedding_experiment.jsonl"


passages = []

with open(DATA_PATH, "r", encoding="utf-8") as file:
    for line in file:
        passage = json.loads(line)
        passages.append(passage)


print(f"Loaded {len(passages)} passages.")


passage_texts = []

for passage in passages:
    passage_texts.append(passage["text"])


load_dotenv()
client = genai.Client()


"""Creating document embeddings"""

passage_response = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=passage_texts,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
)


"""Query embeddings"""

query_response = client.models.embed_content(
    model=EMBEDDING_MODEL,
    contents=QUERY,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
)


"""Converting embeddings into Numpy arrays to calculate cosine similarity"""

passage_embeddings = []

for embedding in passage_response.embeddings:
    passage_embeddings.append(embedding.values)


passage_embeddings = np.array(passage_embeddings)


query_embedding = np.array(query_response.embeddings[0].values)


print(f"Passage embedding matrix shape: {passage_embeddings.shape}")

print(f"Query embedding shape: {query_embedding.shape}")


"""Calculating Cosine similarity for each paragraph"""

similarities = []

for passage_embedding in passage_embeddings:
    dot_product = np.dot(
        query_embedding,
        passage_embedding,
    )

    query_magnitude = np.linalg.norm(query_embedding)

    passage_magnitude = np.linalg.norm(passage_embedding)

    cosine_similarity = dot_product / (query_magnitude * passage_magnitude)

    similarities.append(cosine_similarity)


similarities = np.array(similarities)


"""Ranking Passages"""

ranked_indices = np.argsort(similarities)[::-1]


print("\n")
print("=" * 80)
print("QUERY")
print("=" * 80)

print(QUERY)


print("\n")
print("=" * 80)
print(f"TOP {TOP_K} RETRIEVED PASSAGES")
print("=" * 80)


for rank, index in enumerate(
    ranked_indices[:TOP_K],
    start=1,
):
    passage = passages[index]

    score = similarities[index]

    print(f"\nRANK #{rank}")

    print(f"Similarity: {score:.4f}")

    print(f"Passage ID: {passage['passage_id']}")

    print(f"Matter ID: {passage['matter_id']}")

    print(f"Document: {passage['title']}")

    print(f"Section: {passage['section']}")

    print("\nText:")

    print(passage["text"])

    print("-" * 80)
