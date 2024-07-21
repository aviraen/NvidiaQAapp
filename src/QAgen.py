import openai
from Retrieval import get_top_docs

openai.api_key = "your_api_key"

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
