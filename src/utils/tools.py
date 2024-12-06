import requests
import re
import os

from langchain_core.tools import tool
from serpapi import GoogleSearch
from typing import Union, List
from pinecone import Pinecone, ServerlessSpec

from services.llmprovider import LLMProvider
embeddings = LLMProvider("ollama_embedding")


# Initialize the Pinecone client
pc = Pinecone()

# Define the serverless specification for Pinecone (AWS region 'us-east-1')
# At this moment, only this region is avaialable for the free tier
spec = ServerlessSpec(
    cloud='aws', 
    region='us-east-1'
)

index_name = 'langgraph-research-agent'
index = pc.Index(index_name)


# ------------------------------------------------------
# --- Tool to fetch the abstract from an ArXiv paper ---
# ------------------------------------------------------

# Compile a regular expression pattern to find the abstract in the HTML response
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

@tool('fetch_arxiv')
def fetch_arxiv(arxiv_id: str) -> str:
    '''Fetches the abstract from an ArXiv paper given its ArXiv ID.

    Args:
        arxiv_id (str): The ArXiv paper ID.
    
    Returns:
        str: The extracted abstract text from the ArXiv paper.
    '''

    res = requests.get(f'https://arxiv.org/abs/{arxiv_id}')
    
    re_match = abstract_pattern.search(res.text)

    return re_match.group(1) if re_match else 'Abstract not found.'


# --------------------------------------------------------
# --- Tool for web search with SerpAPI - Google Search ---
# --------------------------------------------------------

# Set up the SerpAPI request parameters, including the API key.
serpapi_params = {
    'engine': 'google',  
    'api_key': os.getenv('SERPAPI_KEY') 
}


@tool('web_search')
def web_search(query: str) -> str:
    '''Finds general knowledge information using a Google search.

    Args:
        query (str): The search query string.
    
    Returns:
        str: A formatted string of the top search results, including title, snippet, and link.
    '''

    search = GoogleSearch({
        **serpapi_params,  
        'q': query,        
        'num': 5         
    })
   
    results = search.get_dict().get('organic_results', [])
    formatted_results = '\n---\n'.join(
        ['\n'.join([x['title'], x['snippet'], x['link']]) for x in results]
    )
    
    # Return the formatted results or a 'No results found.' message if no results exist.
    return formatted_results if results else 'No results found.'



# -------------------------------------------------------------------
# --- Creating RAG Tools for Retrieval-Augmented Generation (RAG) ---
# -------------------------------------------------------------------

def format_rag_contexts(matches: list) -> str:
    '''Formats the retrieved context matches into a readable string format.

    Args:
        matches (list): A list of matched documents with metadata.
    
    Returns:
        str: A formatted string of document titles, chunks, and ArXiv IDs.
    '''
    formatted_results = []
    
    # Loop through each match and extract its metadata.
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Chunk: {x['metadata']['chunk']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
        )
        # Append each formatted string to the results list.
        formatted_results.append(text)
    
    # Join all the individual formatted strings into one large string.
    return '\n---\n'.join(formatted_results)

@tool('rag_search_filter')
def rag_search_filter(query: str, arxiv_id: str) -> str:
    '''Finds information from the ArXiv database using a natural language query and a specific ArXiv ID.

    Args:
        query (str): The search query in natural language.
        arxiv_id (str): The ArXiv ID of the specific paper to filter by.
    
    Returns:
        str: A formatted string of relevant document contexts.
    '''
    
    # Encode the query into a vector representation.
    #xq = encoder([query]) # openai encoder
    xq = embeddings.embed_query(query) # ollama encoder
    
    # Perform a search on the Pinecone index, filtering by ArXiv ID.
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={'arxiv_id': arxiv_id})
    
    # Format and return the search results.
    return format_rag_contexts(xc['matches'])

@tool('rag_search')
def rag_search(query: str) -> str:
    '''Finds specialist information on AI using a natural language query.

    Args:
        query (str): The search query in natural language.
    
    Returns:
        str: A formatted string of relevant document contexts.
    '''
    
    # Encode the query into a vector representation.
    #xq = encoder([query])
    xq = embeddings.embed_query(query) # ollama encoder
    
    # Perform a broader search without filtering by ArXiv ID.
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    
    # Format and return the search results.
    return format_rag_contexts(xc['matches'])


# --------------------------------------------------------------
# Define the 'final_answer' tool to compile the research report.
# --------------------------------------------------------------
@tool
def final_answer(
    introduction: str,
    research_steps: Union[str, List[str]],
    main_body: str,
    conclusion: str,
    sources: Union[str, List[str]],
) -> str:
    '''Returns a natural language response in the form of a research report.

    Args:
        introduction (str): A short paragraph introducing the user's question and the topic.
        research_steps (str or list): Bullet points or text explaining the steps taken for research.
        main_body (str): The bulk of the answer, 3-4 paragraphs long, providing high-quality information.
        conclusion (str): A short paragraph summarizing the findings.
        sources (str or list): A list or text providing the sources referenced during the research.

    Returns:
        str: A formatted research report string.
    '''

    # Format research steps if given as a list.
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    
    # Format sources if given as a list.
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    
    # Construct and return the final research report.
    return f'{introduction}\n\nResearch Steps:\n{research_steps}\n\nMain Body:\n{main_body}\n\n \
    Conclusion:\n{conclusion}\n\nSources:\n{sources}'


# -------------
# --- TOOLS ---
# -------------
tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer
]
