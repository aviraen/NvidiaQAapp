## ![download (1)](https://github.com/user-attachments/assets/f6c57ddc-2621-4d1f-b48e-9dd6666f85e5) ![download (5)](https://github.com/user-attachments/assets/636de63f-5fc6-47fc-9497-35ddfd0200ab)                                                                         ![download (2)](https://github.com/user-attachments/assets/9bd300ba-1258-4117-9282-3322dedd1b42)

# NvidiaQAapp
This project implements a sophisticated question-answering system for NVIDIA CUDA documentation. It includes web crawling, data processing, vector database creation, and a retrieval system integrated with a language model for answering user queries.
## Project Components
1. Web Crawler
2. Data Chunker and Embedder
3. Milvus Vector Database Uploader
4. Retrieval and Re-ranking System
5. Question Answering with LLM
6. User Interface

## Setup Instructions
### Prerequisites
- Python 3.8+
- pip
- Git
### Installation
1. Clone the repository:
2. Create a virtual environment:
3. Install required packages:
4. Set up environment variables:
Create a `.env` file in the project root and add:
ZILLIZ_ENDPOINT=your_zilliz_endpoint
ZILLIZ_TOKEN=your_zilliz_token
OPENAI-API-KEY
## Running the System

### 1. Web Crawling

Run the web crawler to scrape CUDA documentation:
This will crawl https://docs.nvidia.com/cuda/ and its sublinks up to a depth of 5 levels, saving the data to `cuda_documentation.csv`.

### 2. Data Chunking and Embedding

Process the crawled data, chunk it based on semantic similarity, and create embeddings:

This script reads `cuda_documentation.csv`, chunks the content, and saves the result to `chunked_csvfile.csv`.

### 3. Uploading to Milvus

Upload the chunked and embedded data to the Milvus vector database:
This script reads `chunked_data.csv` and uploads the data to the Milvus vectordb collection.

### 4. Retrieval and Re-ranking

To run the retrieval system:
This script implements query expansion, hybrid retrieval (BM25 + BERT), and re-ranking.

### 5. Question Answering

To start the question-answering system:
This integrates the retrieval system with the chosen LLM to answer user queries.

### 6. User Interface (Optional)

implemented, run the user interface:

streamlit run ui.py

