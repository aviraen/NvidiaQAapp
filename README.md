## ![download (1)](https://github.com/user-attachments/assets/f6c57ddc-2621-4d1f-b48e-9dd6666f85e5)                                                                         ![download (2)](https://github.com/user-attachments/assets/9bd300ba-1258-4117-9282-3322dedd1b42)

# CUDA Documentation QA System

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Obtaining Zilliz Cloud Token](#obtaining-zilliz-cloud-token)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Project Overview
This CUDA Documentation QA System is an advanced question-answering tool designed to assist users with queries related to NVIDIA CUDA programming. It leverages a sophisticated retrieval system, combining BM25 and dense vector search, followed by reranking and GPT-3.5 for answer generation.

**Key features:**
- Web-based user interface for easy interaction
- Hybrid retrieval system (BM25 + dense retrieval)
- Cross-encoder reranking for improved relevance
- GPT-3.5 integration for natural language answer generation
- Query history tracking

## Prerequisites
- Python 3.8+
- pip
- Git
- Zilliz Cloud account
- OpenAI API key

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cuda-doc-qa.git
    cd cuda-doc-qa
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Obtaining Zilliz Cloud Token
To use the Zilliz Cloud (Milvus) service:

1. Sign up for a Zilliz Cloud account at [cloud.zilliz.com](https://cloud.zilliz.com).
2. Create a new cluster or use an existing one.
3. In the cluster details, find the "Endpoint" and "Token" information.
4. Copy these values; you'll need them for configuration.

## Configuration

1. Create a `.env` file in the project root:
    ```bash
    touch .env
    ```

2. Add the following to the `.env` file:
    ```plaintext
    ZILLIZ_ENDPOINT=your_zilliz_endpoint
    ZILLIZ_TOKEN=your_zilliz_token
    OPENAI_API_KEY=your_openai_api_key
    ```
    Replace `your_zilliz_endpoint`, `your_zilliz_token`, and `your_openai_api_key` with your actual credentials.

## Crawler Configuration:

[`cuda_crawler.py`](./cuda_crawler.py): The script that performs the actual web scraping.

**Common Crawler Issues:**
- **No Data Retrieved**: Ensure that the target URLs are correct and that the website structure hasn't changed.
- **Connection Errors**: Check your network connection and the target website’s availability.

## Usage

1. Start the Streamlit app:
    ```bash
    streamlit run ui.py
    ```

2. Open a web browser and go to the URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501)).
3. Enter your CUDA-related questions in the text input field and click "Get Answer".
4. View the generated answer and your query history on the interface.

## Project Structure

- [`Retrieval.py`](./retrieval.py): Handles connection to Milvus, query expansion, hybrid retrieval, and reranking.
- [`QAgen.py`](./QAgen.py): Integrates retrieval with GPT-3.5 for answer generation.
- [`UI.py`](./UI.py): Streamlit-based user interface.
- [`requirements.txt`](./requirements.txt): List of Python dependencies.
- [`Chunking.py`](./Chunking.py): Chunk the data using sementic similrity method and embed them using sentence transformer model.
- [`README.md`](./README.md): This file.

## Troubleshooting

**Zilliz Cloud Connection Issues:**
- Ensure your Zilliz Cloud cluster is running.
- Verify that the endpoint and token in `.env` are correct.
- Check your network connection and firewall settings.

**OpenAI API Issues:**
- Confirm your API key is correct and has sufficient credits.
- Check OpenAI's [status page](https://status.openai.com/) for any ongoing service issues.

**Retrieval Problems:**
- Ensure your Milvus collection is properly populated with CUDA documentation data.
- Verify that the collection name in `retrieval.py` matches your Milvus collection name.

**UI Not Loading:**
- Check that Streamlit is installed correctly.
- Ensure all dependencies are installed (`pip install -r requirements.txt`).

**Slow Performance:**
- Consider optimizing your Milvus index settings.
- Adjust the number of documents retrieved and reranked in `retrieval.py`.

For any persistent issues, please check the error messages in the console and refer to the documentation of the respective libraries (Milvus, OpenAI, Streamlit) for more specific troubleshooting steps. For additional support or to report bugs, please open an issue in the GitHub repository.
