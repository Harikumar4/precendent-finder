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

# Initialize or load the ChromaDB collection
def initialize_chroma():
    if not st.session_state.initialized:
        try:
            st.session_state.chroma_client = chromadb.Client()
            try:
                st.session_state.chroma_client.delete_collection(name="precedent_finder")
            except:
                pass  # Ignore if collection doesn't exist
            
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

# Main app
def main():
    st.title("⚖️ Legal Precedent Finder")
    st.markdown("Search through legal precedents using natural language queries")
    
    # Initialize ChromaDB
    if not initialize_chroma():
        st.error("Failed to initialize the database. Please check the logs.")
        return

    # Sidebar for file upload
    with st.sidebar:
        st.header("📂 Upload Case Files")
        uploaded_files = st.file_uploader(
            "Upload PDF case files", 
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            processed = 0
            total_chunks = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                case_id = f"uploaded_{i}"
                file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                chunks, _ = ingest_case(file_path, case_id)
                total_chunks += chunks
                processed += 1
                progress_bar.progress(processed / total_files)
                
                # Clean up
                os.remove(file_path)
            
            if total_chunks > 0:
                st.success(f"Successfully processed {processed} file(s) with {total_chunks} chunks")
            else:
                st.warning("No content was processed. Check if the PDFs contain extractable text.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application helps you search through legal precedents using semantic search.
        Upload PDF case files and search using natural language queries.
        """)

    # Main search interface
    st.markdown("### 🔍 Search Legal Precedents")
    query = st.text_area(
        "Enter your legal query (e.g., 'cases about copyright infringement'):",
        height=100
    )
    
    n_results = st.slider("Number of results to show", 1, 20, 5)
    
    if st.button("Search", type="primary") and query:
        with st.spinner("Searching through legal precedents..."):
            try:
                results = search_cases(query, n_results)
                
                if results and 'documents' in results and results['documents']:
                    total_results = len(results['documents'][0])
                    st.subheader(f"📄 Search Results (showing {min(8, total_results)} of {total_results} matches)")
                    
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
                                ### 📄 Result {i} 
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
                    st.warning("No results found. Try a different query or upload more case files.")
                    
            except Exception as e:
                st.error(f"An error occurred during search: {str(e)}")

if __name__ == "__main__":
    main()
