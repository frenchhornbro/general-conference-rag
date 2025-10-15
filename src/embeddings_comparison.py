import ast
import json
import logging
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils.openai import generate_embeddings_openai

logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('display.max_colwidth', None)

with open("config.json") as config:
  c = json.load(config)
  embeddings_dir = c["embeddings_dir"]

clusters_dir = f"{embeddings_dir}/clusters"
free_dir = f"{embeddings_dir}/free"
openai_dir = f"{embeddings_dir}/openai"

def get_query_embedding(query, embedding_type):
  embedding = None
  if embedding_type == "free":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(
      query,
      batch_size=32,
      show_progress_bar=False,
      convert_to_numpy=True,
      normalize_embeddings=True
    )
  elif embedding_type == "openai":
    embedding = generate_embeddings_openai([query])[0]
  else:
    raise RuntimeError("Type must be specified")
  logging.info("Embedding generated for query")
  return np.array(embedding)

def _compare_embeddings(query_embedding, dir, csv_name, num_results):
  # Get CSV embeddings
  open_file = os.path.join(dir, csv_name)
  df = pd.read_csv(open_file)
  logging.info(f"Data loaded for {dir}/{csv_name}")
  # Compare embeddings
  df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
  embeddings = np.stack(df["embedding"].values)
  df["similarity"] = cosine_similarity([query_embedding], embeddings)[0]
  closest_embeddings = df.nlargest(num_results, 'similarity')
  logging.info(f"{num_results} closest embeddings to {csv_name} found")
  return closest_embeddings

def find_closest(query_embedding, embedding_type, csv_name, num_results=3):
  if embedding_type == "clusters":
    return _compare_embeddings(query_embedding, clusters_dir, csv_name, num_results)
  elif embedding_type == "free":
    return _compare_embeddings(query_embedding, free_dir, csv_name, num_results)
  elif embedding_type == "openai":
    return _compare_embeddings(query_embedding, openai_dir, csv_name, num_results)
  else:
    raise RuntimeError("Type must be specified")

if __name__ == "__main__":
  query = input("Enter your query: ")
  cols = ['title', 'season', 'year', 'text', 'similarity']
  # Using SentenceTransformer embeddings
  free_query_embedding = get_query_embedding(query, "free")
  # From paragraphs
  free_closest_paragraphs = find_closest(free_query_embedding, "free", "paragraph.csv")
  print(f"Closest paragraphs (free):\n{free_closest_paragraphs[cols]}\n")
  # From talks
  free_closest_talks = find_closest(free_query_embedding, "free", "talk.csv")
  print(f"Closest talks (free):\n{free_closest_talks[cols]}\n")
  # From clusters
  free_closest_clusters = find_closest(free_query_embedding, "clusters", "free_3_clusters.csv")
  print(f"Closest clusters (free):\n{free_closest_clusters[cols]}\n")

  # Using OpenAI embeddings
  openai_query_embedding = get_query_embedding(query, "openai")
  # From paragraphs
  openai_closest_paragraphs = find_closest(openai_query_embedding, "openai", "paragraph.csv")
  print(f"Closest paragraphs (openai):\n{openai_closest_paragraphs[cols]}\n")
  # From talks
  openai_closest_talks = find_closest(openai_query_embedding, "openai", "talk.csv")
  print(f"Closest talks (free):\n{openai_closest_talks[cols]}\n")
  # From clusters
  openai_closest_clusters = find_closest(openai_query_embedding, "clusters", "openai_3_clusters.csv")
  print(f"Closest clusters (free):\n{openai_closest_clusters[cols]}\n")