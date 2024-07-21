import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, Collection
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import wordnet
import torch
import openai
import numpy as np
import streamlit as st
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
openai.api_key = "your_api_key"
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

def answer_question(query, context):
    prompt = f"""You are an AI assistant specializing in CUDA programming. 
    Answer the following question based on the provided context. 
    If the answer is not in the context, say "I don't have enough information to answer that question."

    Context: {context}

    Question: {query}

    Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about CUDA programming."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].message['content'].strip()

def process_query(query):
    expanded_query = query_expansion(query)
    retrieved_ids = hybrid_retrieval(expanded_query)
    reranked_ids = rerank(query, retrieved_ids)
    
    top_docs = collection.query(expr=f"id in {reranked_ids[:3]}", output_fields=["content"])
    context = " ".join([doc['content'] for doc in top_docs])
    
    answer = answer_question(query, context)
    
    return answer

def main():
    st.set_page_config(page_title="CUDA Query Assistant", page_icon="🚀", layout="wide")
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

   
    st.sidebar.image("https://developer.nvidia.com/sites/default/files/akamai/cuda/images/cuda_logo.jpg", width=200)
    st.sidebar.title("CUDA Query Assistant")
    st.sidebar.info(
        "This app uses a hybrid retrieval system combining BM25 and dense retrieval, "
        "followed by cross-encoder reranking and GPT-3.5 for answer generation. "
        "It's designed to answer questions about CUDA programming based on the official documentation."
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("Ask Your CUDA Questions")
        query = st.text_input("Enter your CUDA-related question:", key="query_input")
        
        if 'history' not in st.session_state:
            st.session_state.history = []
        if st.button("Get Answer", key="submit_button"):
            if query:
                with st.spinner("Processing your query..."):
                    answer = process_query(query)
                    st.session_state.history.append((query, answer))
                st.success("Answer generated!")
            else:
                st.warning("Please enter a question.")       
        if st.session_state.history:
            st.subheader("Latest Answer")
            st.info(st.session_state.history[-1][1])
    with col2:
        st.subheader("Query History")
        for i, (q, a) in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {q[:50]}{'...' if len(q) > 50 else ''}"):
                st.write(f"A: {a}")
    st.markdown("---")
    st.markdown("Developed with ❤️ by Your Team | Powered by Streamlit")
if __name__ == "__main__":
    main()