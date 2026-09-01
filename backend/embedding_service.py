from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings.tolist()


def generate_embedding(text):
    embeddings = generate_embeddings([text])
    return embeddings[0]