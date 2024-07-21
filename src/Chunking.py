import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(text, model):
    return model.encode(text)

df = pd.read_csv('processed_file.csv')

def chunk_text(df, model, max_chunk_size=500):
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    current_chunk_embedding = np.zeros(384)

    for _, row in df.iterrows():
        content = row['content']
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_embedding = get_embeddings(sentence, model)
            sentence_length = len(sentence.split())
            similarity = cosine_similarity([current_chunk_embedding], [sentence_embedding])[0][0]
            
            if current_chunk_size + sentence_length > max_chunk_size or similarity < 0.5:
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'urls': row['url'],
                    'titles': row['title']
                })
                current_chunk = [sentence]
                current_chunk_size = sentence_length
                current_chunk_embedding = sentence_embedding
            else:
                current_chunk.append(sentence)
                current_chunk_size += sentence_length
                current_chunk_embedding = np.mean([current_chunk_embedding, sentence_embedding], axis=0)

    if current_chunk:
        chunks.append({
            'content': ' '.join(current_chunk),
            'urls': df.iloc[-1]['url'],
            'titles': df.iloc[-1]['title']
        })

    return pd.DataFrame(chunks)
chunked_df = chunk_text(df, model)
chunked_df.to_csv('chunked_csvfile.csv', index=False)

