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
    top_docs = get_top_docs(query)
    context = " ".join(top_docs)
    answer = answer_question(query, context)
    return answer

if __name__ == "__main__":
    test_query = "What is the difference between global and shared memory in CUDA?"
    answer = process_query(test_query)
    print(f"Question: {test_query}")
    print(f"Answer: {answer}")
