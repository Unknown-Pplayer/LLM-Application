from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import getpass
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from scipy.spatial.distance import cosine
import numpy as np
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def extract_keywords(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Extract entities and unique nouns as keywords
    entities = [entity.text for entity in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    # Combine entities and nouns, and filter out duplicates
    keywords = list(set(entities + nouns))
    return keywords

# Example usage
question = "What is Professor McGonagall's role during the first-years' arrival at Hogwarts, and how does she interact with them?"
question2 = "what is role of MCGONAGALL"
keywords = extract_keywords(question)
print(keywords)

load_dotenv()
embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")

def calculate_cosine_similarity(embedding1, embedding2):
    # Using the scipy cosine function, subtract from 1 to get similarity
    return 1 - cosine(embedding1, embedding2)

#  Simulate embedding queries from database, which contains a list of questions
q_embeddings = [
    embedder.embed_query("who is McGonagall?"),
    embedder.embed_query("What is Professor McGonagall's role during the first-years' arrival at Hogwarts, and how does she interact with them?"),
    embedder.embed_query("what is 1 + 1"),
]
questions = [
    "who is McGonagall?",
    "What is Professor McGonagall's role during the first-years' arrival at Hogwarts, and how does she interact with them?", 
    "what is 1 + 1"
]

#  Simulate a specific question query
question = "what is role of MCGONAGALL?"

#  Embed the specific question and calculate similarities
specific_q_embedding = embedder.embed_query(question)
similarities = [calculate_cosine_similarity(specific_q_embedding, q_emb) for q_emb in q_embeddings]
most_similar_index = np.argmax(similarities)
print(f"Most similar question index: {most_similar_index}")
print(f"Similarity score: {similarities[most_similar_index]}")
print("Most similar question:", questions[most_similar_index])
for i, similarity_score in enumerate(similarities):
    print(f"Question: {questions[i]}")
    print(f"Similarity score: {similarity_score}\n")


model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "what is role of MCGONAGALL",
    "who is McGonagall?",
    "What is Professor McGonagall's role during the first-years' arrival at Hogwarts, and how does she interact with them?", 
    "what is 1 + 1"
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)