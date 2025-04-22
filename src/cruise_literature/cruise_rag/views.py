from pyexpat.errors import messages
import time
from django.http import Http404
from django.shortcuts import get_object_or_404, render

# Create your views here.

from elasticsearch import Elasticsearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..citation_screening.views import create_screening_decisions, move_paper_to_done

from ..citation_screening.models import CitationScreening

from ..literature_review.models import LiteratureReview

# Initialize Elasticsearch and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="langchain_index",
    embedding=embeddings,
)


def add_paper_to_elasticsearch_index(review_id, paper):
    """
    Function to index embeddings.
    """
    print("Adding paper to Elasticsearch index for RAG...")
    print(f"Paper: {paper}")
    print(f"Review ID: {review_id}")
    print()
