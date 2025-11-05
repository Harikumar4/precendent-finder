import streamlit as st
import os
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Legal Precedent Finder",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
    st.session_state.collection = None
    st.session_state.initialized = False
    st.session_state.cases_loaded = False

# Initialize or load the ChromaDB collection
def initialize_chroma():
    if not st.session_state.initialized:
        try:
            # Use PersistentClient to save embeddings to disk
            db_path = "./chroma_db"
            st.session_state.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Check if collection already exists
            try:
                st.session_state.collection = st.session_state.chroma_client.get_collection(
                    name="precedent_finder",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
                # Collection exists, check if it has data
                count = st.session_state.collection.count()
                if count > 0:
                    st.session_state.cases_loaded = True
            except:
                # Collection doesn't exist, create it
                st.session_state.collection = st.session_state.chroma_client.create_collection(
                    name="precedent_finder",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                )
            
            st.session_state.initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
            return False
    return True

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to parse metadata
def parse_metadata(text):
    metadata = {}
    title_match = re.search(r"Case Details\s*\n([\s\S]*?)\n\n", text)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    judges_match = re.search(r"\[(.*?)JJ?\.\]", text)
    if judges_match:
        judges = [j.strip().replace("*","") for j in judges_match.group(1).split("and")]
        metadata["judges"] = ", ".join(judges)

    keywords_match = re.search(r"List of Keywords\n(.*?)\n\n", text, re.DOTALL)
    if keywords_match:
        metadata["keywords"] = ", ".join([k.strip() for k in keywords_match.group(1).split(";")])

    acts_match = list(set(re.findall(r"Article \d+|IPC \d+|CrPC \d+", text)))
    metadata["sections"] = ", ".join(acts_match)

    year_match = re.search(r"\[(\d{4})\]", text)
    if year_match:
        metadata["year"] = int(year_match.group(1))

    metadata["court"] = "Supreme Court of India"
    return metadata

# Function to chunk text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Increased for more context
        chunk_overlap=200,  # Increased for better context between chunks
        separators=["\n\n", "\n", ". ", " ", ""]  # Added more separators
    )
    return splitter.split_text(text)

# Function to ingest a case
def ingest_case(pdf_path, case_id):
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        metadata = parse_metadata(raw_text)
        chunks = chunk_text(raw_text)

        for i, chunk in enumerate(chunks):
            st.session_state.collection.add(
                documents=[chunk],
                metadatas=[{
                    **metadata,
                    "case_id": case_id,
                    "chunk_id": i
                }],
                ids=[f"{case_id}_{i}"]
            )
        return len(chunks), metadata
    except Exception as e:
        st.error(f"Error ingesting {case_id}: {str(e)}")
        return 0, {}

# Function to load cases from cases/ folder
def load_cases_from_folder(folder_path="cases", max_cases=10):
    """Load the first N PDF cases from the cases folder into ChromaDB"""
    if not st.session_state.initialized:
        return False
    
    # Check if collection already has data
    try:
        count = st.session_state.collection.count()
        if count > 0:
            st.session_state.cases_loaded = True
            return True
    except:
        pass
    
    if st.session_state.cases_loaded:
        return True
    
    try:
        cases_path = Path(folder_path)
        if not cases_path.exists():
            st.error(f"Cases folder '{folder_path}' not found!")
            return False
        
        # Get all PDF files, sorted by name
        pdf_files = sorted([f for f in cases_path.glob("*.pdf")])[:max_cases]
        
        if not pdf_files:
            st.warning(f"No PDF files found in '{folder_path}' folder!")
            return False
        
        total_chunks = 0
        progress_bar = st.progress(0)
        total_files = len(pdf_files)
        
        for idx, pdf_file in enumerate(pdf_files):
            case_id = pdf_file.stem  # e.g., "2024-1-case-1"
            chunks, _ = ingest_case(str(pdf_file), case_id)
            total_chunks += chunks
            progress_bar.progress((idx + 1) / total_files)
        
        st.session_state.cases_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading cases: {str(e)}")
        return False

# Function to search cases
def search_cases(query, n_results=8):  # Increased default results
    try:
        # Enhanced search with distances for relevance scoring
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Add relevance scores to metadata
        if results and 'distances' in results and results['distances']:
            for i, dist in enumerate(results['distances'][0]):
                if results['metadatas'][0][i]:
                    results['metadatas'][0][i]['relevance'] = 1.0 - (dist / 2.0)  # Convert distance to 0-1 score
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

# Function to generate answer using Groq API
def generate_answer_with_groq(query, chunks, metadatas, api_key, model="llama-3.1-70b-versatile"):
    """
    Generate an answer using Groq API based on the query and relevant chunks.
    
    Args:
        query: User's query
        chunks: List of relevant text chunks
        metadatas: List of metadata for each chunk
        api_key: Groq API key
        model: Groq model to use (default: llama-3.1-70b-versatile)
    
    Returns:
        Generated answer string or None if error
    """
    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Format chunks with their metadata for context
        context_parts = []
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
            context_info = f"[Source {i}]"
            if 'title' in metadata:
                context_info += f" Case: {metadata['title']}"
            if 'case_id' in metadata:
                context_info += f" (ID: {metadata['case_id']})"
            if 'year' in metadata:
                context_info += f" Year: {metadata['year']}"
            context_info += f"\n{chunk}\n"
            context_parts.append(context_info)
        
        # Combine all context
        context = "\n\n".join(context_parts)
        
        # Create the prompt
        system_prompt = """You are a legal research assistant specializing in Indian case law. 
Your task is to provide clear, accurate, and well-structured answers based on the provided legal case excerpts.
Use the source information to support your answer and cite specific cases when relevant.
If the provided context doesn't fully answer the question, state what information is available and what is missing."""
        
        user_prompt = f"""Based on the following legal case excerpts, please answer the query: "{query}"

Legal Case Excerpts:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the query
2. References specific cases and excerpts when relevant
3. Provides a clear, structured response
4. Cites the source information (Source 1, Source 2, etc.) when making specific points

Answer:"""
        
        # Call Groq API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating answer with Groq: {str(e)}")
        return None

# Main app
def main():
    st.title("⚖️ Legal Precedent Finder")
    st.markdown("Search through legal precedents using natural language queries")
    
    # Initialize ChromaDB
    if not initialize_chroma():
        st.error("Failed to initialize the database. Please check the logs.")
        return
    
    # Load cases from cases/ folder automatically (only if not already loaded)
    if not st.session_state.cases_loaded:
        with st.spinner("Creating embeddings from cases/ folder (this only happens once)..."):
            if load_cases_from_folder("cases", max_cases=10):
                count = st.session_state.collection.count()
                st.success(f"✅ Loaded 10 cases from cases/ folder ({count} chunks total)")
            else:
                st.error("Failed to load cases. Please check the cases/ folder.")
    else:
        # Show info about existing embeddings
        count = st.session_state.collection.count()
        st.success(f"✅ Using existing embeddings ({count} chunks)")

    # Groq API Configuration (always enabled)
    # Try to get API key from environment first
    env_api_key = os.environ.get("GROQ_API_KEY")
    
    if env_api_key:
        st.session_state.groq_api_key = env_api_key
    else:
        # Store in session state for access in main function
        if 'groq_api_key' not in st.session_state:
            st.session_state.groq_api_key = None
        
        # API key input at the top
        groq_api_key_input = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.session_state.groq_api_key if st.session_state.groq_api_key else "",
            help="Enter your Groq API key. Get one at https://console.groq.com/ or set GROQ_API_KEY environment variable"
        )
        
        if groq_api_key_input:
            st.session_state.groq_api_key = groq_api_key_input
    
    # Fixed settings
    groq_model = "llama-3.1-8b-instant"  # Always use this model
    chunks_for_answer = 5  # Always use top 5 chunks

    # Main search interface
    st.markdown("### 🔍 Search Legal Precedents")
    query = st.text_area(
        "Enter your legal query:",
        height=100,
        placeholder="e.g., 'cases about copyright infringement' or 'trivial errors in recruitment applications'"
    )
    
    # Fixed settings
    groq_model = "llama-3.1-8b-instant"  # Always use this model
    chunks_for_answer = 5  # Always use top 5 chunks
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching through legal precedents..."):
            try:
                # Get top 5 chunks for Groq
                results = search_cases(query, n_results=5)
                
                if results and 'documents' in results and results['documents']:
                    total_results = len(results['documents'][0])
                    
                    # Get Groq API key from session state or environment
                    groq_api_key = st.session_state.get('groq_api_key') or os.environ.get("GROQ_API_KEY")
                    
                    # Generate AI answer using Groq (always enabled)
                    ai_answer = None
                    if groq_api_key:
                        # Use top 5 chunks for answer generation
                        chunks_for_ai = results['documents'][0][:5]
                        metadatas_for_ai = results['metadatas'][0][:5]
                        
                        with st.spinner(f"🤖 Generating AI answer using {groq_model}..."):
                            ai_answer = generate_answer_with_groq(
                                query, 
                                chunks_for_ai, 
                                metadatas_for_ai,
                                groq_api_key,
                                groq_model
                            )
                    else:
                        st.warning("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable or enter it in the sidebar.")
                    
                    # Display AI answer if available
                    if ai_answer:
                        st.markdown("---")
                        st.markdown(ai_answer)
                        st.markdown("---")
                    
                    # Display search results (top 5 chunks)
                    st.subheader(f"📄 Source Chunks (Top 5)")
                    
                    for i, (doc, metadata, dist) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results.get('distances', [[0]*len(results['documents'][0])])[0]
                    ), 1):
                        with st.container():
                            # Calculate relevance score (1 - normalized distance)
                            relevance_score = 1.0 - (dist / 2.0)  # Convert to 0-1 scale
                            relevance_color = "green" if relevance_score > 0.7 else "orange" if relevance_score > 0.4 else "red"
                            
                            st.markdown(f"""
                                ### 📄 Chunk {i} 
                                <span style="color: {relevance_color}; font-size: 0.8em;">
                                    Relevance: {relevance_score:.0%}
                                </span>
                            """, unsafe_allow_html=True)
                            
                            # Display metadata
                            metadata_display = []
                            if 'title' in metadata:
                                metadata_display.append(f"**Title:** {metadata['title']}")
                            if 'case_id' in metadata:
                                metadata_display.append(f"**Case ID:** {metadata['case_id']}")
                            if 'judges' in metadata:
                                metadata_display.append(f"**Judges:** {metadata['judges']}")
                            if 'year' in metadata:
                                metadata_display.append(f"**Year:** {metadata['year']}")
                            if 'sections' in metadata:
                                metadata_display.append(f"**Sections:** {metadata['sections']}")
                            if 'keywords' in metadata:
                                metadata_display.append(f"**Keywords:** {metadata['keywords']}")
                                
                            st.markdown("  \n".join(metadata_display))
                            
                            # Display document excerpt with more context
                            st.markdown("**Relevant Excerpt:**")
                            st.markdown(f"> {doc[:1000]}{'...' if len(doc) > 1000 else ''}")
                            
                            st.markdown("---")
                else:
                    st.warning("No results found. Make sure cases are loaded in the database.")
                    
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")

if __name__ == "__main__":
    main()
