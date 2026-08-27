import nest_asyncio
nest_asyncio.apply()

# chatbot.py — NASA Space Biology RAG Chatbot (English, Open Source LLM)

import os, re, time, tempfile
from functools import lru_cache
import random 

import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup

# --- RAG/LLM Dependencies ---
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.prompts import PromptTemplate

# -------------------------
# Utility: Star Generation Logic (CSS for background)
# -------------------------
def generate_star_shadows(n_stars, max_w_vw=100, max_h_vh=200, color='#FFF', size=1.0):
    """Generates a long CSS box-shadow string for star fields."""
    shadows = []
    for _ in range(n_stars):
        x = random.randint(1, max_w_vw)
        y = random.randint(1, max_h_vh)
        s = random.random() * size
        shadows.append(f"{x}vw {y}vh {s}px {s}px {color}")
    return ',\n'.join(shadows)

stars_layer_1_shadows = generate_star_shadows(700, size=0.1)
stars_layer_2_shadows = generate_star_shadows(200, size=0.5, color='#EEE')
stars_layer_3_shadows = generate_star_shadows(50, size=1.0, color='#DDD')

# -------------------------
# Page config & style (Space Theme with Animation - English UI)
# -------------------------
st.set_page_config(page_title="BioNovaX Chatbot", layout="wide", page_icon="icon.png")


# Inject the hidden div for the third, slow layer of stars
st.markdown("<div id='star-layer-3'></div>", unsafe_allow_html=True)

# -------------------------
# Settings & paths
# -------------------------
NCBI_API_KEY = os.getenv("05d60bc6977f33cb391be15db599e79c4e09", "") 
EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CSV_PATH = "SB_publication_PMC.csv"
VECTOR_DB_DIR = "chroma_db"
# --- IMPORTANT: CHANGE THIS PATH TO YOUR DOWNLOADED GGUF FILE NAME ---
LLM_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" 
# ------------------------------------------------------------------------

# Caching for PMC XML content
CACHE_DIR = "pmc_cache"
os.makedirs(CACHE_DIR, exist_ok=True) 

# -------------------------
# Load dataset & Models (Cached)
# -------------------------
@st.cache_data(show_spinner="Loading and preparing dataset...")
def load_dataset(path=CSV_PATH):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["idx"] = df.index.astype(int)
    return df

@st.cache_resource(show_spinner="Loading Embedder Model (HuggingFace)...")
def load_embedder_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource(show_spinner=f"Loading Local LLM (from {LLM_MODEL_PATH})...")
def load_local_llm(model_path):
    if not os.path.exists(model_path):
         st.error(f"LLM GGUF file not found at: {model_path}. Please download it and restart.")
         st.stop()
         
    return LlamaCpp(
        model_path=model_path,
        temperature=0.0, 
        n_ctx=4096, 
        n_gpu_layers=-1, # -1 attempts to use all GPU layers
        verbose=False,
    )

# -------------------------
# NCBI EFetch & Parsing Helpers 
# -------------------------
def extract_pmcid(link):
    if not isinstance(link, str): return None
    m = re.search(r"(PMC\d+)", link, re.IGNORECASE)
    return m.group(1) if m else None

@lru_cache(maxsize=2048)
def efetch_pmc_xml(pmcid: str):
    cache_file = os.path.join(CACHE_DIR, f"{pmcid}.xml")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read(), "CACHE"
    
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    if NCBI_API_KEY: params["api_key"] = NCBI_API_KEY
    r = requests.get(EFETCH_BASE, params=params, timeout=30)
    r.raise_for_status()
    xml_text = r.text
    time.sleep(0.34)
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(xml_text)
    return xml_text, "EFETCH"

def parse_full_text(xml_text):
    soup = BeautifulSoup(xml_text, "xml")
    text_parts = []
    
    # 1. Abstract
    ab = soup.find("abstract")
    if ab:
        text_parts.extend([p.get_text(" ", strip=True) for p in ab.find_all("p")])

    # 2. Main Body Sections 
    for sec in soup.find_all("sec"):
        paras = [p.get_text(" ", strip=True) for p in sec.find_all("p")]
        text_parts.extend(paras)
    
    # 3. Fallback for main body
    if not text_parts:
        b = soup.find("body")
        if b:
            text_parts.extend([p.get_text(" ", strip=True) for p in b.find_all("p")])
            
    full_text = " ".join(text_parts).strip()
    return full_text if len(full_text) > 200 else None 

# -------------------------
# RAG Pipeline Core Functions - WITH st.status LOADER
# -------------------------

# FIX: Changed 'embed_model' to '_embed_model' to resolve UnhashableParamError
@st.cache_resource(show_spinner=False) 
def build_vector_store(df, _embed_model):
    
    # 1. Check if DB already exists
    if os.path.exists(VECTOR_DB_DIR):
        st.success("Vector Store already exists. Loading database...")
        # Use _embed_model here
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embed_model)

    # Use st.status for a descriptive, multi-step loader on the initial run
    with st.status("🛠️ **Setting up RAG Knowledge Base (Initial Run)**", expanded=True) as status:
        
        st.write("Step 1/3: Fetching and parsing full text content from PMC...")
        
        texts = []
        subset_df = df.head(50) # Processing subset for quick test
        
        # Inner Progress Bar for fetching loop
        progress_bar = st.progress(0, text=f"Processing {len(subset_df)} articles...")
        
        for i, row in subset_df.iterrows():
            pmcid = extract_pmcid(row.get("Link", ""))
            full_text = None
            if pmcid:
                try:
                    xml, _ = efetch_pmc_xml(pmcid)
                    full_text = parse_full_text(xml)
                except Exception:
                    pass 

            if full_text and len(full_text) > 200:
                meta = {"source": row["Title"], "link": row["Link"], "pmc": pmcid}
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len
                )
                chunks = text_splitter.create_documents([full_text], [meta])
                texts.extend(chunks)
            
            progress_bar.progress((i + 1) / len(subset_df))

        progress_bar.empty()
        st.info(f"Indexed {len(texts)} chunks of scientific content.")

        # 3. Create ChromaDB
        status.update(label="Step 2/3: Generating and indexing vector embeddings...", state="running", expanded=True)
        if not texts:
             st.error("No valid PMC content could be fetched or parsed for RAG indexing.")
             return None
             
        # Use _embed_model here
        vectorstore = Chroma.from_documents(
            texts,
            _embed_model, 
            persist_directory=VECTOR_DB_DIR
        )
        
        status.update(label="Step 3/3: Persisting database to disk...", state="running", expanded=True)
        vectorstore.persist()
        
        status.update(label="✅ RAG Knowledge Base successfully built!", state="complete", expanded=False)
        return vectorstore

# -------------------------
# Main Application Flow
# -------------------------

# Initial Checks
if not os.path.exists(CSV_PATH):
    st.error(f"Dataset CSV file not found at: `{CSV_PATH}`.")
    st.stop()
if not os.path.exists(LLM_MODEL_PATH):
    st.error(f"Local LLM GGUF file not found at: `{LLM_MODEL_PATH}`. Please download it.")
    st.stop()

# Load Resources 
df = load_dataset()
embed_model = load_embedder_model()
llm = load_local_llm(LLM_MODEL_PATH)

# Start RAG Indexing process (will display the st.status loader on first run)
# Note: The embed_model is passed here, but the function signature handles the unhashable error.
vectorstore = build_vector_store(df, embed_model) 

# --- Define the RAG Chain ---
qa_chain = None
if vectorstore:
    # Prompt engineering for analytical, English-only responses
    SYSTEM_PROMPT = """You are an expert Space Biology Knowledge Engine. 
    Your role is to analyze scientific findings, synthesize complex information, 
    and answer questions based ONLY on the provided context (scientific abstracts and full-text excerpts). 
    Do not use outside knowledge. If the context does not contain the answer, state 'The provided context does not contain a sufficient answer.'.
    All your responses MUST be in English.
    """
    
    prompt_template = SYSTEM_PROMPT + "\n\nCONTEXT: {context}\n\nQUESTION: {question}"
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    st.sidebar.success("RAG System is Ready!")
else:
    st.sidebar.error("RAG system could not be initialized.")


st.title("🚀 NASA Space Biology RAG Chatbot")
st.write("An Open Source Deep Learning engine that synthesizes and analyzes scientific literature (PMC articles) in English.")

# --- Chat Interface Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about Microgravity, Radiation, or Space Biology..."):
    
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if qa_chain is None:
        response = "The RAG system failed to load. Please check the logs and the LLM model path."
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # 2. Call the RAG chain
        with st.spinner("Analyzing scientific data and generating response..."):
            
            try:
                result = qa_chain({"query": prompt}) 
                
                response_text = result['result'].strip()
                
                # Extract and format sources
                sources = {}
                for doc in result['source_documents']:
                    title = doc.metadata.get('source', 'Unknown Source')
                    link = doc.metadata.get('link', '#')
                    # Use unique title as key to prevent duplicate sources
                    if title not in sources:
                        sources[title] = link
                
                source_markdown = "\n\n**— Sources —**\n"
                if sources:
                    for i, (title, link) in enumerate(sources.items()):
                        source_markdown += f"- [{i+1}. {title}]({link})\n"
                else:
                    source_markdown += "No specific sources retrieved for this answer."
                    
                final_response = response_text + source_markdown
                
            except Exception as e:
                final_response = f"An error occurred during LLM generation: {e}"
                st.exception(e) 
        
        # 3. Display assistant response
        with st.chat_message("assistant"):
            st.markdown(final_response)
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})