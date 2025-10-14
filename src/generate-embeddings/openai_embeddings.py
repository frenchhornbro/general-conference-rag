import json
import pandas as pd
from datetime import datetime
import os
from openai import OpenAI
import tiktoken

with open("config.json") as config:
  c = json.load(config)
  raw_dir = c["raw_dir"]
  embeddings_dir = c["embeddings_dir"]
  openai_key = c["openAIKey"]

OpenAI.api_key = openai_key
client = OpenAI(api_key=OpenAI.api_key)

def generate_embeddings(csv_file, column_name, output_dir, model="text-embedding-3-small", max_tokens=300_000):
  os.makedirs(output_dir, exist_ok=True)
  try:
    df = pd.read_csv(csv_file)
    texts = df['text'].tolist()

    # Initialize tokenizer
    encoder = tiktoken.encoding_for_model(model)

    # Clean texts and calculate token counts
    text = [text.replace("\n", " ") for text in texts]
    token_counts = [len(encoder.encode(text)) for text in texts]

    embeddings = []
    current_batch = []
    current_token_count = 0

    for i, (text, token_count) in enumerate(zip(texts, token_counts)):
      if current_token_count + token_count > max_tokens or len(current_batch) > 100:
        # Process current batch
        response = client.embeddings.create(input=current_batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        # Reset batch
        current_batch = [text]
        current_token_count = token_count
      else:
        current_batch.append(text)
        current_token_count += token_count

    # Process final batch
    if current_batch:
      response = client.embeddings.create(input=current_batch, model=model)
      batch_embeddings = [item.embedding for item in response.data]
      embeddings.extend(batch_embeddings)

    # Save embeddings to disk
    df['embedding'] = embeddings
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