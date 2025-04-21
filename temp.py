from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

import getpass
import os
import dotenv
import json

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# es = Elasticsearch([{"host": "127.0.0.1", "port": 9200, "scheme": "http"}],)
# es._verified_elasticsearch = True

elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    # es_connection = es,
    index_name="langchain_index",
    embedding=embeddings,
    es_user="elastic",
    es_password="changeme",
)

mytext = "This is a test text to be embedded and stored in Elasticsearch."
# Store the text in Elasticsearch
elastic_vector_search.add_texts([mytext])
# Retrieve the text from Elasticsearch
retrieved_texts = elastic_vector_search.similarity_search(mytext)
for text in retrieved_texts:
    print(text)

PATH = "scripts/data/tmp/dblpv13.jsonl"
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

with open(PATH, "r") as f:
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
        
            # loader = TextLoader("data/my_corpus.txt")
            # docs = loader.load()

            # chunks = splitter.split_documents(docs)