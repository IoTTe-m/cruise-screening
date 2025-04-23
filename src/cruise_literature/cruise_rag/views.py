
# Create your views here.
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from django.core.files.temp import NamedTemporaryFile
import requests


# Initialize Elasticsearch and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def review_id_to_index(review_id):
    return f'review_{review_id}_index'


def download_pdf(pdf_url):
    print(f"Downloading PDF from {pdf_url}")
    try:
        temp_file = NamedTemporaryFile(delete=True)
        response = requests.get(pdf_url)
        if response.status_code == 200:
            temp_file.write(response.content)
            temp_file.flush()
            print(f"PDF downloaded to {temp_file.name}")
            return temp_file
        else:
            print(f"Failed to download PDF: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None


def add_paper_to_elasticsearch_index(review_id, paper):
    """
    Function to index embeddings.
    """
    print("Adding paper to Elasticsearch index for RAG...")
    # print(f"Paper: {paper}")
    print(f"PDF: {paper['pdf'] if 'pdf' in paper else 'No PDF'}")
    print(f"Review ID: {review_id}")
    print()
    # we only care about documents with pdf
    # we download the pdf
    # we convert create embedings 
    # we will use the pdf as the document if it exists, otherwise we will use the abstract
    # we will have to add appropriate metadata to the document, so that we can use it to provide citations

    paper_pages = []

    if 'pdf' in paper and paper['pdf']:
        pdf_url = paper['pdf']
        temp_pdf_file = download_pdf(pdf_url)
        if temp_pdf_file:
            # Load the PDF and extract text
            pdf_loader = PyPDFLoader(temp_pdf_file.name)
            paper_pages = pdf_loader.load_and_split()

            temp_pdf_file.close()

            for paper_page in paper_pages:
                paper_page.metadata['title'] = paper['title']
                paper_page.metadata['authors'] = paper['authors']
                paper_page.metadata['doi'] = paper['doi']
                paper_page.metadata['abstract'] = paper['abstract']

            print(f"Loaded {len(paper_pages)} pages from PDF.")
        else:
            print("No PDF file found or failed to download.")
    else:
        print("No PDF file found, using abstract instead.")
        print(paper)
        paper_pages = [
            Document(
                page_content=paper['abstract'],
                metadata={
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'doi': paper['doi'],
                    'abstract': paper['abstract'],
                    'page': -1,
                }
            )
        ]

    index_name = review_id_to_index(review_id)
    elastic_vector_search = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
    )
    elastic_vector_search.add_documents(paper_pages)

def remove_paper_from_elasticsearch_index(review_id, paper):
    pass
