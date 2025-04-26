# import os
# from sentence_transformers import SentenceTransformer
# import faiss
# import json

# MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# EMBED_DIM = 384

# DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
# INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
# DOCS_PATH = os.path.join(DATA_DIR, 'docs.json')


# def load_docs():
#     with open(DOCS_PATH, 'r', encoding='utf-8') as f:
#         return json.load(f)


# def build_index():
#     docs = load_docs()
#     texts = [doc['text'] for doc in docs]
#     model = SentenceTransformer(MODEL_NAME)
#     embeddings = model.encode(texts, show_progress_bar=True)

#     index = faiss.IndexFlatIP(EMBED_DIM)
#     index.add(embeddings)
#     os.makedirs(DATA_DIR, exist_ok=True)
#     faiss.write_index(index, INDEX_PATH)
#     print(f"✅ FAISS index saved to {INDEX_PATH}")


# if __name__ == '__main__':
#     build_index()






import os
import torch
import json
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

MODEL_NAME = 'facebook/dpr-ctx_encoder-single-nq-base'
EMBED_DIM = 768

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
INDEX_PATH = os.path.join(DATA_DIR, 'faiss.index')
DOCS_PATH = os.path.join(DATA_DIR, 'docs.json')

def load_docs():
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_index():
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(MODEL_NAME)
    model = DPRContextEncoder.from_pretrained(MODEL_NAME)
    model.eval()

    docs = load_docs()
    texts = [doc['text'] for doc in docs]

    # Batched processing for large inputs
    embeddings = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            batch_embeddings = model(**inputs).pooler_output
            embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    os.makedirs(DATA_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index rebuilt with DPR and saved to {INDEX_PATH}")

if __name__ == '__main__':
    build_index()
