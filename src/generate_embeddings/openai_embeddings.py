import json
import pandas as pd
from datetime import datetime
import os
from utils.openai import generate_embeddings_openai

with open("config.json") as config:
  c = json.load(config)
  raw_dir = c["raw_dir"]
  embeddings_dir = c["embeddings_dir"]

def generate_embeddings(csv_file, column_name, output_dir, model="text-embedding-3-small", max_tokens=300_000):
  os.makedirs(output_dir, exist_ok=True)
  try:
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()
    # Save embeddings to disk
    df['embedding'] = generate_embeddings_openai(texts)
    output_file = os.path.join(output_dir, f"{column_name}.csv")
    df.to_csv(output_file, index=False)
  except Exception as e:
    print(f"An error occurred: \033[35m{e}\033[0m")

if __name__ == "__main__":
  timestamp = datetime.now()
  print("Start talks: ", datetime.now().strftime("%H:%M:%S"))
  generate_embeddings(f"{raw_dir}/SCRAPED_TALKS.csv", "talk", f"{embeddings_dir}/openai")
  print("Start paragraphs: ", datetime.now().strftime("%H:%M:%S"))
  generate_embeddings(f"{raw_dir}/SCRAPED_PARAGRAPHS.csv", "paragraph", f"{embeddings_dir}/openai")
  print("Finish: ", datetime.now().strftime("%H:%M:%S"))