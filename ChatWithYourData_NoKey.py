#from openai import OpenAI
import openai
import streamlit as st
import os
import streamlit as st
import chardet
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Remove proxy settings if they exist
if "http_proxy" in os.environ:
    del os.environ["http_proxy"]

if "https_proxy" in os.environ:
    del os.environ["https_proxy"]


# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
#my_key = <YOUR KEY>
my_key = openai.api_key

openai.api_key = my_key
# Print the OpenAI API key
print(f'OpenAI API Key: {openai.api_key}')

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Read documents from the directory
documents_dir = "documents"
documents = []
filenames = []
for filename in os.listdir(documents_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(documents_dir, filename)
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            documents.append(file.read())
            filenames.append(filename)

# Step 1: Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

# Step 2: Build a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(doc_vectors)

def retrieve_relevant_docs(query, k=2):
    query_vector = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

def generate_response(query, include_docs):
    # Retrieve relevant documents
    context = ""

    if include_docs == "Yes":
        relevant_docs = retrieve_relevant_docs(query)
        context = " ".join(relevant_docs)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    include_docs = st.selectbox("Include related documents in context?", ["No", "Yes"])
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        openai.api_key = my_key
        print(f'OpenAI API Key has been set: {openai_api_key}')


    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)




    msg = generate_response(prompt, include_docs)
    #msg = response.choices[0].message['content']
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

