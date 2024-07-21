import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import wordnet
import numpy as np
import torch

ZILLIZ_ENDPOINT = "https://in03-60169b2a1d6a8d8.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_TOKEN = ""

print("Connecting to Zilliz Cloud...")
connections.connect(
    alias="default", 
    uri=ZILLIZ_ENDPOINT, 
    token=ZILLIZ_TOKEN,
    secure=True
)
print("Connected successfully!")

collection = Collection("cuda_docs_embeddings")
collection.load()

sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def query_expansion(query):
    tokens = nltk.word_tokenize(query)
    pos_tags = nltk.pos_tag(tokens)
    
    expanded_query = query
    for word, pos in pos_tags:
        if pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'):
            synsets = wordnet.synsets(word)
            if synsets:
                synonym = synsets[0].lemmas()[0].name()
                if synonym != word:
                    expanded_query += f" OR {synonym}"
    
    return expanded_query

def hybrid_retrieval(query, top_k=100):
    all_docs = collection.query(expr="", output_fields=["content"], limit=collection.num_entities)
    bm25 = BM25Okapi([doc['content'] for doc in all_docs])
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_k = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    query_embedding = sentence_transformer.encode(query)
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 0}
    dense_results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "content"]
    )
    combined_results = set(bm25_top_k + [hit.id for hit in dense_results[0]])
    return list(combined_results)

def rerank(query, doc_ids, top_k=10):
    docs = collection.query(expr=f"id in {doc_ids}", output_fields=["id", "content"])
    pairs = [[query, doc['content']] for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    id_score_pairs = list(zip(doc_ids, scores))
    id_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return [id for id, _ in id_score_pairs[:top_k]]

def get_top_docs(query, num_docs=3):
    expanded_query = query_expansion(query)
    retrieved_ids = hybrid_retrieval(expanded_query)
    reranked_ids = rerank(query, retrieved_ids)
    
    top_docs = collection.query(expr=f"id in {reranked_ids[:num_docs]}", output_fields=["content"])
    return [doc['content'] for doc in top_docs]

if __name__ == "__main__":
    test_query = "How does CUDA handle memory management?"
    top_docs = get_top_docs(test_query)
    print(f"Top documents for query '{test_query}':")
    for i, doc in enumerate(top_docs, 1):
        print(f"{i}. {doc[:200]}...")
