
# Create your views here.
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


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
