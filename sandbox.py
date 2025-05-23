from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import dotenv


dotenv.load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

elastic_vector_search = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="test_index",
    embedding=embeddings,
)

document = Document(
    page_content="This is a test document.",
    metadata={
        "title": "Test Document",
        "authors": "John Doe",
        "doi": "10.1234/test",
        "abstract": "This is a test abstract.",
        "page": 0,
    },
)
paper_pages = [document]

elastic_vector_search.add_documents(paper_pages)
