## ![download (1)](https://github.com/user-attachments/assets/f6c57ddc-2621-4d1f-b48e-9dd6666f85e5)                                                                         ![download (2)](https://github.com/user-attachments/assets/9bd300ba-1258-4117-9282-3322dedd1b42)

# CUDA Documentation QA System
## Table of Contents

Project Overview
Prerequisites
Installation
Obtaining Zilliz Cloud Token
Configuration
Usage
Project Structure
Troubleshooting

## Project Overview
This CUDA Documentation QA System is an advanced question-answering tool designed to assist users with queries related to NVIDIA CUDA programming. It leverages a sophisticated retrieval system, combining BM25 and dense vector search, followed by reranking and GPT-3.5 for answer generation.
Key features:

Web-based user interface for easy interaction
Hybrid retrieval system (BM25 + dense retrieval)
Cross-encoder reranking for improved relevance
GPT-3.5 integration for natural language answer generation
Query history tracking

## Prerequisites

1. Python 3.8+
2. pip
3. Git
4. Zilliz Cloud account
5. OpenAI API key

# Obtaining Zilliz Cloud Token
### To use the Zilliz Cloud (Milvus) service:

Sign up for a Zilliz Cloud account at cloud.zilliz.com.
Create a new cluster or use an existing one.
In the cluster details, find the "Endpoint" and "Token" information.
Copy these values; you'll need them for configuration.

## Project Structure

Retrieval.py: Handles connection to Milvus, query expansion, hybrid retrieval, and reranking.
question_answering.py: Integrates retrieval with GPT-3.5 for answer generation.
UI.py: Streamlit-based user interface.
requirements.txt: List of Python dependencies.
.env: Configuration file for environment variables (not tracked by Git).
README.md: This file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aviraen/NvidiaQAapp.git
cd cuda_crawler

2. Create and activate a virtual environment:
python -m venv venv

3. Install required packages
pip install -r requirements.txt





