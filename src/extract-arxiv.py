import os
import requests
import time
import pandas as pd
import json
import logging
import xml.etree.ElementTree as ET #parsing and manipulating XML data

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pinecone import Pinecone, ServerlessSpec
from tqdm.autonotebook import tqdm

from services.llmprovider import LLMProvider




# https://info.arxiv.org/help/api/basics.html
ARXIV_NAMESPACE = '{http://www.w3.org/2005/Atom}'


def resolve_and_create_folder(relative_folder_path):
    """
    Resolves a folder path relative to the script's directory, creates the folder if it doesn't exist,
    and returns the absolute path.

    Args:
        relative_folder_path (str): Path of the folder relative to the script directory.

    Returns:
        str: Absolute path of the resolved folder.
    """
    # Get the directory of the script being executed.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve the full path for the folder relative to the script directory.
    full_folder_path = os.path.join(script_dir, relative_folder_path)
    
    # Ensure the directory exists.
    os.makedirs(full_folder_path, exist_ok=True)
    
    return full_folder_path


def extract_from_arxiv(search_query='cat:cs.AI', max_results=100, folder_name='files', json_file_name='arxiv_dataset.json'):
    """
    Fetches papers from the ArXiv API based on a search query, saves them as JSON, 
    and returns a pandas DataFrame.

    Args:
        search_query (str): The search query for ArXiv (default is 'cat:cs.AI').
        max_results (int): The maximum number of results to retrieve (default is 100).
        folder_name (str): Name of the folder where the JSON file will be saved.
        json_file_name (str): Name of the JSON file to save the data.

    Returns:
        pd.DataFrame: DataFrame containing the extracted paper information.
    """
    # Resolve the folder path and ensure it exists.
    folder_path = resolve_and_create_folder(folder_name)

    # Combine folder and file name to get the full JSON file path.
    json_file_path = os.path.join(folder_path, json_file_name)

    # Construct the URL for the API request.
    url = f'http://export.arxiv.org/api/query?search_query={search_query}&max_results={max_results}'
    
    # Send a GET request to the ArXiv API.
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for HTTP requests that failed.

    # Parse the XML response.
    root = ET.fromstring(response.content)
      
    papers = []
    
    # Loop through each "entry" in the XML, representing a single paper.
    for entry in root.findall(f'{ARXIV_NAMESPACE}entry'):
        title = entry.find(f'{ARXIV_NAMESPACE}title').text.strip()
        summary = entry.find(f'{ARXIV_NAMESPACE}summary').text.strip()

        # Get the authors of the paper.
        author_elements = entry.findall(f'{ARXIV_NAMESPACE}author')
        authors = [author.find(f'{ARXIV_NAMESPACE}name').text for author in author_elements]

        # Get the paper's URL.
        paper_url = entry.find(f'{ARXIV_NAMESPACE}id').text
        arxiv_id = paper_url.split('/')[-1]

        # Check for the PDF link.
        pdf_link = next((link.attrib['href'] for link in entry.findall(f'{ARXIV_NAMESPACE}link') 
                         if link.attrib.get('title') == 'pdf'), None)

        papers.append({
            'title': title,
            'summary': summary,
            'authors': authors,
            'arxiv_id': arxiv_id,
            'url': paper_url,
            'pdf_link': pdf_link
        })
    
    # Convert list into a pandas DataFrame.
    df = pd.DataFrame(papers)
    
    # Save the DataFrame to a JSON file.
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)
        logging.info(f"Data saved to {json_file_path} ...")

    return df


def download_pdfs(df, folder_name='files'):
    """
    Downloads PDFs from URLs listed in the DataFrame and saves them to a specified folder. 
    The file names are stored in a new column 'pdf_file_name' in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'pdf_link' column with URLs to download.
        download_folder (str): Path to the folder where PDFs will be saved (default is 'files').
    
    Returns:
        pd.DataFrame: The original DataFrame with an additional 'pdf_file_name' column containing 
                      the paths of the downloaded PDF files or None if the download failed.
    """
    
    # Resolve the folder path and ensure it exists.
    download_folder = resolve_and_create_folder(folder_name)
    

    pdf_file_names = []
    
    # Loop through each row to download PDFs
    for index, row in df.iterrows():
        pdf_link = row['pdf_link']
        
        try:
            response = requests.get(pdf_link)
            response.raise_for_status()
    
            file_name = os.path.join(download_folder, pdf_link.split('/')[-1]) + '.pdf'
            pdf_file_names.append(file_name)
    
            # Save the downloaded PDF
            with open(file_name, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"PDF downloaded successfully and saved as {file_name} ...")
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download the PDF: {e}")
            pdf_file_names.append(None)
    
    df['pdf_file_name'] = pdf_file_names

    return df


def load_and_chunk_pdf(pdf_file_name, chunk_size=512):
    """
    Loads a PDF file and splits its content into chunks of a specified size.

    Args:
        file (str): Path to the PDF file to be loaded.
        chunk_size (int): The maximum size of each chunk in characters (default is 512).

    Returns:
        List[Document]: A list of document chunks.
    """

    logging.info(f'Loading and splitting into chunks: {pdf_file_name}')

    # Load the content of the PDF
    loader = PyPDFLoader(pdf_file_name)
    data = loader.load()

    # Split the content into chunks with slight overlap to preserve context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=64)
    chunks = text_splitter.split_documents(data)

    return chunks


def expand_df(df):
    """
    Expands each row in the DataFrame by splitting PDF documents into chunks.

    Args:
        df (pd.DataFrame): DataFrame containing 'pdf_file_name', 'arxiv_id', 'title', 'summary', 
                           'authors', and 'url' columns.

    Returns:
        pd.DataFrame: A new DataFrame where each row represents a chunk of the original document, 
                      with additional metadata such as chunk identifiers and relationships to 
                      adjacent chunks.
    """

    expanded_rows = []  # List to store expanded rows with chunk information

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():
        try:
            chunks = load_and_chunk_pdf(row['pdf_file_name'])
        except Exception as e:
            logging.error(f"Error processing file {row['pdf_file_name']}: {e}")
            continue

        # Loop over the chunks and construct a new DataFrame row for each
        for i, chunk in enumerate(chunks):
            prechunk_id = i-1 if i > 0 else ''  # Preceding chunk ID
            postchunk_id = i+1 if i < len(chunks) - 1 else ''  # Following chunk ID

            expanded_rows.append({
                'id': f"{row['arxiv_id']}#{i}",  # Unique chunk identifier
                'title': row['title'],
                'summary': row['summary'],
                'authors': row['authors'],
                'arxiv_id': row['arxiv_id'],
                'url': row['url'],
                'chunk': chunk.page_content,  # Text content of the chunk
                'prechunk_id': '' if i == 0 else f"{row['arxiv_id']}#{prechunk_id}",  # Previous chunk ID
                'postchunk_id': '' if i == len(chunks) - 1 else f"{row['arxiv_id']}#{postchunk_id}"  # Next chunk ID
            })

    # Return a new expanded DataFrame
    return pd.DataFrame(expanded_rows)


def create_pinecone_index(index_name='langgraph-research-agent', dims=1536):
    # Initialize the Pinecone client
    pc = Pinecone()

    # Define the serverless specification for Pinecone (AWS region 'us-east-1')
    # At this moment, only this region is avaialable for the free tier
    spec = ServerlessSpec(
        cloud='aws', 
        region='us-east-1'
    )


    # Check if the index exists; create it if it doesn't
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=dims,  # Embedding dimension (1536)
            metric='cosine',
            spec=spec  # Cloud provider and region specification
        )

        # Wait until the index is fully initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # Connect to the index
    index = pc.Index(index_name)
    
    # Add a short delay before checking the stats
    #time.sleep(1)

    # View the index statistics
    #index.describe_index_stats()
    
    return index



def upsert_pinecone(data, index, batch_size=64):
    """
    Uploads data to a Pinecone index in batches with a progress bar.

    Args:
        data (pd.DataFrame): DataFrame containing the data to upload.
        index (PineconeIndex): The Pinecone index to upload to.
        batch_size (int): Number of records to process per batch.

    Returns:
        dict: Statistics of the Pinecone index after upsert.
    """
    # Loop through the data in batches, using tqdm for a progress bar
    for i in tqdm(range(0, len(data), batch_size), desc="Uploading to Pinecone"):
        i_end = min(len(data), i + batch_size)  # Define batch endpoint
        batch = data.iloc[i:i_end].to_dict(orient='records')  # Slice data into a batch

        # Extract metadata for each chunk in the batch
        metadata = [{
            'arxiv_id': r['arxiv_id'],
            'title': r['title'],
            'chunk': r['chunk'],
        } for r in batch]
        
        # Generate unique IDs for each chunk
        ids = [r['id'] for r in batch]
        
        # Extract the chunk content
        chunks = [r['chunk'] for r in batch]
        
        # Convert chunks into embeddings
        embeds = embeddings.embed_documents(chunks)
        
        # Upload embeddings, IDs, and metadata to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))
    
    # Display the index statistics after all uploads.
    return index.describe_index_stats()



if __name__ == '__main__':
    logging.info("--- Starting ---")

    # Initialize embeddings and get dimensions
    embeddings = LLMProvider("ollama_embedding")
    encoder = embeddings.embed_query("testing embedding")
    dims = len(encoder)

    logging.info("Extracting data from ArXiv into a Pandas DataFrame and saving it as Json")
    df = extract_from_arxiv(max_results=20) # Extract data from ArXiv

    logging.info("Downloading and Expanding PDF files")
    df = download_pdfs(df) # Download the PDF files
    expanded_df = expand_df(df) # Loading and Splitting PDF Files into Chuncks and Expanding the DataFrame

    logging.info("Uploading to Pinecone")
    index = create_pinecone_index(index_name="langgraph-research-agent", dims=dims) # Building a Knowledge Base for the RAG System Using Embedding
    stats = upsert_pinecone(expanded_df, index)
    logging.info(f"Index stats: {stats}") # View the index statistics after all uploads.stats
    
    logging.info("--- Finished ---")