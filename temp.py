from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import getpass
import os
import dotenv
import json

dotenv.load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="langchain_index",
    embedding=embeddings,
    # es_user="elastic",
    # es_password="changeme",
)

# indices = elastic_vector_search.client.cat.indices()
# elastic_vector_search.client.indices.delete(index="langchain_index", ignore=[400, 404])

# PATH = "scripts/data/tmp/dblpv13_1000.jsonl"
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# with open("data/my_corpus.txt", "w") as f:
#     f.write("")

# with open(PATH, "r") as f:
#     with open("data/my_corpus.txt", "a") as fout:
#         documents = []
        
#         for line in f:
#             line_json = json.loads(line)
#             title = line_json.get("title")
#             abstract = line_json.get("abstract")
#             url = line_json.get("url")
#             if title and abstract and url:
#                 fout.write(f"{title}\n{abstract}\n")
            

        
# loader = TextLoader("data/my_corpus.txt")
# docs = loader.load()
# chunks = splitter.split_documents(docs)
# elastic_vector_search.add_documents(chunks)

# mytext = "gaming"
# retrieved_texts = elastic_vector_search.similarity_search(mytext, k=10)
# print(retrieved_texts)

from langchain.tools import tool
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
)

rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
    retriever=elastic_vector_search.as_retriever(search_kwargs={"k": 3})
)

@tool
def rag_tool(query: str) -> str:
    """Use this tool to get information from the RAG model if needed."""
    return rag_chain.invoke(query)

tools = [rag_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5,
    early_stopping_method="generate",
    verbose=True,
)

query = "Do you know any game-like advertising systems?"
result = agent.invoke(query)
print(result)

#TODO: system prompt
