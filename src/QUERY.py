import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ZILLIZ_ENDPOINT = "https://in03-60169b2a1d6a8d8.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_TOKEN =""
print("Connecting to Zilliz Cloud...")
connections.connect(
    alias="default", 
    uri=ZILLIZ_ENDPOINT, 
    token=ZILLIZ_TOKEN,
    secure=True
)
print("Connected successfully!")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535),  
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=65535), 
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="CUDA documentation embeddings")
collection_name = "cuda_docs_embeddings"
if collection_name in utility.list_collections():
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)
chunked_df = pd.read_csv(r'C:\Users\Admin\cuda_crawler\DATA\chunked_csvfile.csv')
def get_embeddings(text):
    if isinstance(text, str):
        return model.encode(text).tolist()
    else:
        raise ValueError("Input text is not a string")
urls = chunked_df['urls'].astype(str).tolist()  
titles = chunked_df['titles'].astype(str).tolist()  
contents = chunked_df['content'].astype(str).tolist()  
embeddings = [get_embeddings(content) for content in contents]
entities = [
    urls,
    titles,
    contents,
    embeddings
]
collection.insert(entities)
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
print("Chunked data with embeddings inserted, indexed, and loaded into Milvus.")
