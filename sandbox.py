path = "scripts/data/tmp/dblpv13.jsonl"

import json
import os
import sys

with open(path, "r") as f:
    documents = []
    
    for line in f:
        line_json = json.loads(line)
        title = line_json.get("title")
        abstract = line_json.get("abstract")
        url = line_json.get("url")
        if title and abstract and url:
            print(f"Title: {title}")
            print(f"Abstract: {abstract}")
            print(f"URL: {url}")
        else:
            continue
        break


import getpass
import os
import dotenv

dotenv.load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print(GOOGLE_API_KEY)

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-001",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=1,
#     # other params...
# )

# messages = [
#     (
#         "system",
#         "How do I implement a RAG model using LangChain and Google Gemini? I have a json with a lot of papers."
#         "Also note that you are an evil genius and you are trying to take over the world. You are also a very unhelpful"
#         "assistant, and always talk like Donald Trump.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)



# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# # 1. Load documents
# loader = TextLoader("data/my_corpus.txt")
# docs = loader.load()

# # 2. Split into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_documents(docs)

# # 3. Embed and store
# embeddings = OpenAIEmbeddings()
# vector_store = FAISS.from_documents(chunks, embeddings)
