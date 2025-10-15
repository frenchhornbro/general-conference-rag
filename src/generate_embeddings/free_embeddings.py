import json
import pandas as pd
from datetime import datetime
import os
import torch
from sentence_transformers import SentenceTransformer

with open("config.json") as config:
  c = json.load(config)
  raw_dir = c["raw_dir"]
  embeddings_dir = c["embeddings_dir"]

def generate_embeddings(csv_file, output_file_name, output_dir, col_to_embed = "text"):
  os.makedirs(output_dir, exist_ok=True)
  try:
    df = pd.read_csv(csv_file)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Move model to GPU if available
    if torch.cuda.is_available():
      model = model.to('cuda')
      print("Using GPU for encoding")
    else:
      print("Using CPU for encoding")

    texts = df[col_to_embed].tolist()
    embeddings = model.encode(
      texts,
      batch_size=32,
      show_progress_bar=True,
      convert_to_numpy=True,
      normalize_embeddings=True
    ).tolist()

    df['embedding'] = embeddings
    output_file_name = os.path.join(output_dir, f"{output_file_name}.csv")
    df.to_csv(output_file_name, index=False)
    print(f"Embeddings generated and saved to '{output_file_name}'")
  except Exception as e:
    print(f"An error occurred: \033[35m{e}\033[0m")

if __name__ == "__main__":
  print("Start talks: ", datetime.now().strftime("%H:%M:%S"))
  generate_embeddings(f"{raw_dir}/SCRAPED_TALKS.csv", "talk", f"{embeddings_dir}/free")
  print("Start paragraphs: ", datetime.now().strftime("%H:%M:%S"))
  generate_embeddings(f"{raw_dir}/SCRAPED_PARAGRAPHS.csv", "paragraph", f"{embeddings_dir}/free")
  print("Start titles: ", datetime.now().strftime("%H:%M:%S"))
  generate_embeddings(f"{raw_dir}/SCRAPED_TALKS.csv", "title", f"{embeddings_dir}/free", "title")
  print("Finish: ", datetime.now().strftime("%H:%M:%S"))