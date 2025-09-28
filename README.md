# Legal Precedent Finder

A powerful tool for searching and analyzing legal precedents using natural language processing and semantic search.

## Features

- **Semantic Search**: Find relevant legal cases using natural language queries
- **PDF Processing**: Extract and index text from legal documents
- **Metadata Extraction**: Automatically extract case details like title, judges, year, and legal sections
- **Relevance Scoring**: Results are ranked by relevance with visual indicators
- **User-Friendly Interface**: Clean, intuitive web interface built with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GriffinJolly/precedent-finder.git
   cd precedent-finder
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv precedent_finder_env
   # On Windows:
   .\precedent_finder_env\Scripts\activate
   # On macOS/Linux:
   # source precedent_finder_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to upload PDF case files

4. Enter your search query in the main text area and click "Search"

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `main.ipynb`: Jupyter notebook with initial development code
- `script.ipynb`: Additional utility scripts
- `cases/`: Directory for storing case PDFs

## Dependencies

### Core Dependencies
- Python 3.8+
- PyMuPDF==1.23.26
- sentence-transformers==2.7.0
- langchain==0.2.11
- chromadb==0.5.0

### Data Processing
- pandas==2.2.2
- numpy==1.26.4

### Web Interface
- streamlit==1.36.0

### Full list
See `requirements.txt` for complete list of dependencies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Griffin Jolly](https://github.com/GriffinJolly)
