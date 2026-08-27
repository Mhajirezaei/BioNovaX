<<<<<<< HEAD
# BioNovaX
BioNovaX is an intelligent platform by BioNova Family that uses advanced NLP and interactive knowledge graphs to structure NASA space biology research, reveal knowledge gaps, and deliver actionable insights faster to researchers, managers, and mission planners.
=======
# 🚀 NASA Space Biology Knowledge Engine

This project comprises two Python applications that form an advanced knowledge engine for exploring and analyzing NASA Space Biology publications. It combines **online data extraction (NCBI PMC)**, **advanced AI summarization (Pegasus)**, and **knowledge graph visualization (NetworkX/Pyvis)** to quickly uncover scientific relationships and research gaps.

## Features

### 1\. Main Knowledge Engine (`app.py`)

A comprehensive Streamlit dashboard providing targeted insights on Space Biology publications.

  * **Targeted Search:** Search the local publication database by title keyword.
  * **Real-time Data Fetching:** Automatically fetches the full article content (Abstract, Results, Discussion, Conclusion, Body) from **NCBI PubMed Central (PMC)** using the EFetch API.
  * **Entity Extraction:** Identifies key Space Biology concepts (e.g., Microgravity, Bone Loss, Radiation) from the article text using predefined keywords.
  * **AI Summarization:** Uses a **Seq2Seq model (Google's Pegasus-XSum)** to generate concise, abstractive summaries of the most relevant article section (configurable to Abstract, Results, or Conclusion).
  * **Knowledge Graph (KG):** Builds a dynamic, interactive graph (using **NetworkX/Pyvis**) showing the connections between articles (nodes) and the biological **entities** they discuss, highlighting co-occurrence relationships.
  * **Data Identifier Search:** Extracts raw data repository IDs (e.g., **GLDS, GEO, SRA, PRIDE**) mentioned in the full text, linking the research to the underlying omics data.
  * **Meta-Summarization:** Generates a single, high-level comprehensive summary by integrating the summaries of all returned search results.
  * **Downloadable Insights:** Export all extracted data, summaries, and links to a CSV file.

### 2\. Companion RAG Chatbot (`rag_chatbot_app.py` - Incomplete)

A planned Retrieval-Augmented Generation (RAG) system using an open-source Large Language Model (LLamaCpp) for deep, contextual question-answering over the dataset.

  * **RAG Architecture:** Utilizes **LangChain** and **Chroma** (VectorStore) with **HuggingFaceEmbeddings** for efficient semantic search.
  * **Open-Source LLM:** Designed to use a local **LlamaCpp** model for privacy and cost-efficiency.
  * **Contextual Chat:** Allows users to ask complex questions, with the chatbot grounding its answers in the factual content of the Space Biology publications.
  * **Integration:** Designed to be accessed as a floating popup/iframe within the main `app.py` dashboard for unified interaction.

## Setup and Installation

### Prerequisites

  * Python 3.9+
  * A CSV file named `SB_publication_PMC.csv` containing at least `Title` and `Link` columns, which must include links to full-text articles (preferably PMC links).
  * A model directory containing the required open-source LLM files (e.g., `.gguf` for LlamaCpp), if you plan to run the RAG Chatbot.

### Environment Setup

1.  **Clone the Repository (or save the files):**

    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```

2.  **Install Dependencies:**
    The project uses standard libraries and specific AI/NLP packages.

    ```bash
    pip install pandas streamlit requests beautifulsoup4 transformers scikit-learn sentence-transformers networkx pyvis plotly

    # For the RAG Chatbot (rag_chatbot_app.py):
    pip install langchain chromadb llama-cpp-python nest-asyncio
    ```

3.  **Set Environment Variables:**
    The main application can benefit from an NCBI API key to avoid rate limits when fetching full articles.

    ```bash
    export NCBI_API_KEY="YOUR_NCBI_API_KEY"
    ```

    *(Note: The provided code includes a placeholder API key that should be replaced or set via the environment.)*

## Usage

### 1\. Launch the Main Knowledge Engine

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

### 2\. Launch the RAG Chatbot (Optional/Parallel)

The main application expects the RAG Chatbot to be running on port `8502`. To enable the chatbot modal:

1.  Ensure you have set up the RAG dependencies and model file (if the complete `rag_chatbot_app.py` is available).

2.  Run the chatbot application in a **separate terminal window**:

    ```bash
    streamlit run chatbot.py --server.port 8502
    ```

3.  Once both are running, click the **"⏫"** button in the main app to interact with the integrated RAG Chatbot.

## Code Overview

| File | Description | Key Technologies |
| :--- | :--- | :--- |
| `app.py` | The main Streamlit dashboard. Handles search, NCBI fetching, AI summarization, KG building, and visualization. | Streamlit, requests/BeautifulSoup, transformers (Pegasus), NetworkX/Pyvis, Plotly |
| `rag_chatbot_app.py` | Companion script for the RAG chatbot. Sets up the vector store and retrieval chain for contextual Q\&A. *(Note: Code snippet is incomplete, but outlines RAG setup)* | Streamlit, LangChain, Chroma, HuggingFaceEmbeddings, LlamaCpp |
| `SB_publication_PMC.csv` | Input dataset containing article titles and links (required to run `app.py`). | Data Source |

## Knowledge Graph Structure

The knowledge graph is a **bipartite graph** designed around articles and entities:

  * **Article Nodes (Blue):** Represent individual publications. Their size increases based on the number of entities they discuss.
  * **Entity Nodes (Orange/Cyan):** Represent key Space Biology concepts (e.g., Microgravity, Muscle Atrophy). Their size increases based on the number of articles discussing them.
  * **DESCRIBES Edges (Grey):** Connect an Article node to an Entity node if the article discusses that entity.
  * **CO\_OCCURRENCE Edges (Orange Dashed):** Connect two Entity nodes if they are discussed together in at least one article. Edge weight indicates the frequency of co-occurrence, highlighting strong scientific relationships.
>>>>>>> 4576198 (Initial commit for BioNovaX)
