# Streamlit Chatbot with OpenAI and FAISS

This project is a Streamlit-based chatbot that uses OpenAI's GPT-4 for generating responses and FAISS for efficient document retrieval. The chatbot can include relevant documents in its responses based on the user's query.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- An OpenAI API key. You can get one by signing up at [OpenAI](https://platform.openai.com/signup/).

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```bash
    pip install openai streamlit faiss-cpu numpy scikit-learn chardet
    ```

3. Set up your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your_openai_api_key'
    ```

## Usage

1. Place your text documents in a directory named `documents`. Each document should be a `.txt` file.

2. Run the Streamlit app:

    ```bash
    streamlit run script.py
    ```

3. Open your web browser and go to `http://localhost:8501` to interact with the chatbot.

## Code Explanation

### Import Libraries

```python
import openai
import streamlit as st
import os
import chardet
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer