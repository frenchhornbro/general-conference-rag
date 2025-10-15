import json
from openai import OpenAI
from embeddings_comparer import get_query_embedding, find_closest

with open("config.json") as config:
  c = json.load(config)
  openAIKey = c["openAIKey"]

# Setup OpenAI client
openAIClient = OpenAI(api_key=openAIKey)
openAIClient.models.list() # check the key is valid

def get_chat_gpt_response(query, talk_text):
  chat_query = f"""
  Answer the query given only these paragraphs. Do not use any other context, and do not vary from what the information provided here.
  Do not reference the paragraphs themselves in your response. Limit the responses to <= 100 words.

  Paragraphs:
  \"\"\"
  {talk_text}
  \"\"\"

  Query: 
  \"\"\"
  {query}
  \"\"\"
  """
  stream = openAIClient.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": chat_query}],
    stream=True
  )
  responseList = []
  for chunk in stream:
    if chunk.choices[0].delta.content is not None:
      responseList.append(chunk.choices[0].delta.content)
  return "".join(responseList)

if __name__ == "__main__":
  # Get related talks
  query = input("Enter your query: ")
  query_embedding = get_query_embedding(query, "free")
  free_closest_paragraphs = find_closest(query_embedding, "free", "talk.csv")
  talk_text = "\n".join(free_closest_paragraphs["text"].values.astype(str))
  # Get answer from ChatGPT
  response = get_chat_gpt_response(query, talk_text)
  print(f"Response: {response}")