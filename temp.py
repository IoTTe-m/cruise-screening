from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch

import getpass
import os
import dotenv

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

es = Elasticsearch([{"host": "127.0.0.1", "port": 9200, "scheme": "http"}],)
es._verified_elasticsearch = True

elastic_vector_search = ElasticsearchStore(
    # es_url="http://localhost:9200",
    es_connection = es,
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

print("a")