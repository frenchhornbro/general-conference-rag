import json
from openai import OpenAI
import tiktoken

with open("config.json") as config:
  c = json.load(config)
  embeddings_dir = c["embeddings_dir"]
  openai_key = c["openAIKey"]

OpenAI.api_key = openai_key
client = OpenAI(api_key=OpenAI.api_key)

def generate_embeddings_openai(texts, model="text-embedding-3-small", max_tokens=300_000):
  """
  Parameters:
  - texts: list of strings

  Returns:
  - List of embedding vectors
  """
  encoder = tiktoken.encoding_for_model(model)
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
  return embeddings