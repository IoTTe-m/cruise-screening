
# Create your views here.
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from django.core.files.temp import NamedTemporaryFile
import requests
from sympy import content


# Initialize Elasticsearch and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def review_id_to_index(review_id):
    return f'review_{review_id}_index'

def get_id_for_a_page_by_metadata(title, page, doi):
    return f"{title}_{page}_{doi}"

def get_id_for_a_page(paper_page):
    return get_id_for_a_page_by_metadata(
        paper_page.metadata['title'],
        paper_page.metadata['page'],
        paper_page.metadata['doi'],
    )


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

    try:
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
    except Exception as e:
        print(f"Error loading PDF: {e}")

    print(paper)
    
    if len(paper_pages) == 0:
        print("No PDF file found, using abstract instead.")

        content = paper['abstract'] if 'abstract' in paper else None
        
        if not content:
            print("No abstract found.")
            content = paper['snippet'] if 'snippet' in paper else None

        if not content:
            print("No snippet found, stop.")
            return

        paper_pages = [
            Document(
                page_content=content,
                metadata={
                    'title': paper['title'] if 'title' in paper else "",
                    'authors': paper['authors'] if 'authors' in paper else "",
                    'doi': paper['doi'] if 'doi' in paper else "",
                    'abstract': content,
                    'page': -1,
                }
            )
        ]

    ids = [get_id_for_a_page(paper_page) for paper_page in paper_pages]

    index_name = review_id_to_index(review_id)
    elastic_vector_search = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
    )
    elastic_vector_search.add_documents(paper_pages, ids=ids)

def remove_paper_from_elasticsearch_index(review_id, paper):
    print(f"Removing paper from Elasticsearch index for RAG...")
    index_name = review_id_to_index(review_id)
    elastic_vector_search = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
    )



    # for page_number in range(-1, -2):
    #     page_id = get_id_for_a_page(
