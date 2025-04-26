import os
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

MODEL_NAME = 'facebook/rag-token-base'


from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_from_disk

import os
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# def init_rag():
#     # Exactly match the folder your create_docs.py wrote to
#     passages_dir = os.path.join("data", "processed", "hf_dataset")
#     index_file   = os.path.join("data", "processed", "faiss.index")

#     tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
#     retriever = RagRetriever.from_pretrained(
#         "facebook/rag-sequence-nq",
#         index_name="custom",
#         passages_path=passages_dir,
#         index_path=index_file
#     )
#     model = RagSequenceForGeneration.from_pretrained(
#         "facebook/rag-sequence-nq",
#         retriever=retriever
#     )
#     return tokenizer, model

def init_rag():
    # Point to the folder your create_docs.py actually created:
    dataset_path = os.path.join("data", "processed", "hf_dataset")
    index_path   = os.path.join("data", "processed", "faiss.index")

    # Load the dataset from disk
    dataset = load_from_disk(dataset_path)

    # Initialize retriever against that dataset + FAISS index
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path
    )

    # Load tokenizer and model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model     = RagSequenceForGeneration.from_pretrained(
        "facebook/rag-sequence-nq",
        retriever=retriever
    )

    return tokenizer, model


    



def chat_loop(tokenizer, model):
    print("Welcome to the RAG Chatbot! (type 'exit' to quit)")
    while True:
        query = input("You: ")
        if query.lower() in ('exit', 'quit'):
            break
        inputs = tokenizer([query], return_tensors='pt')
        generated = model.generate(**inputs)
        answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        print(f"Bot: {answer}\n")


if __name__ == '__main__':
    tok, mod = init_rag()
    chat_loop(tok, mod)