# app.py — NASA Space Biology Knowledge Engine (Entity-Centric, Final & Robust)
import os, re, time, tempfile
from functools import lru_cache

import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from pyvis.network import Network
import plotly.express as px

import streamlit.components.v1 as components




st.set_page_config(page_title="BioNovaX", layout="wide", page_icon="icon.png")
st.markdown("""
<style>
/* 1. Global Variables & Dark Space Background */
:root {
    --primary-color: #59a6ff; /* Blue for progress/links/cards highlight */
    --secondary-color: #ff7f50; /* Orange for entity accents */
    --background-color: #0b1220; /* Deep Space Blue/Black (Main Content) */
    --sidebar-bg-color: #070e1a; /* Darker than main for sidebar */
    --text-color: #FFFFFF; /* Light Blue-White for main text and titles */
    --card-bg-start: #071022;
    --card-bg-end: #07182b;
    --card-border: rgba(255,255,255,0.08);
}
body {
    background: var(--background-color);
    color: var(--text-color) !important;
}

/* Star/Dust Effect (Fixed and subtle) */
body::before, body::after {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 200vw;
    height: 200vh;
    background: transparent url("https://www.transparenttextures.com/patterns/stardust.png") repeat;
    animation: moveStars 100s linear infinite;
    opacity: 0.2;
    z-index: -1;
    pointer-events: none;
}

body::after {
    animation-duration: 200s;
    opacity: 0.1;
}

@keyframes moveStars {
    from { transform: translate(0, 0); }
    to { transform: translate(-100vw, -100vh); }
}
.stApp {
  background-color: transparent;
}

/* -------------------- Headings -------------------- */
h1, h2, h3, h4, h5, h6, p, span {
  color: #FFFFFF !important;
  word-wrap: break-word;
}

/* -------------------- Cards -------------------- */
.card {
  background: linear-gradient(180deg,#071022 0%, #07182b 100%);
  border-radius: 15px;
  padding: 16px;
  margin-bottom: 12px;
  border:1px solid rgba(255,255,255,0.06);
  box-shadow: 0 0 20px rgba(89,166,255,0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  word-break: break-word;
  color: #FFFFFF !important;
}
.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 0 25px rgba(89,166,255,0.4);
}

/* -------------------- Small text & links -------------------- */
.small { color: #FFFFFF !important; font-size: 0.9rem; word-wrap: break-word; }
.a-link { color: #FFFFFF !important; text-decoration: none; word-break: break-word; }
.a-link:hover { text-decoration: underline; }

/* -------------------- Sidebar -------------------- */
[data-testid="stSidebar"] {
  background-color: #172135;
  color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stTextInput, 
[data-testid="stSidebar"] .stNumberInput, 
[data-testid="stSidebar"] .stSelectbox, 
[data-testid="stSidebar"] .stCheckbox {
  color: #FFFFFF !important;
}
/* -------------------- Tables -------------------- */
.stDataFrame div[data-testid="stDataFrameContainer"] {
  color: #FFFFFF !important;
}

/* -------------------- Plotly charts -------------------- */
.js-plotly-plot .plotly .main-svg {
  background-color: transparent !important;
}
/* 5. Responsiveness */
@media (max-width: 768px) {
    /* Adjust padding for small screens */
    .st-emotion-cache-1v06a4z { 
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Smaller main title on mobile */
    h1 {
        font-size: 2rem;
    }
    /* Compact card padding */
    .card {
        padding: 15px;
    }
    /* Ensure the Pyvis graph scrolls if too large */
    div[data-testid="stHtml"] iframe {
        min-height: 500px;
    }
}
/* -------------------- Download Button Style -------------------- */

.stDownloadButton > button {
    background-color: #59a6ff; 
    color: #0b1220 !important;
    border: 1px solid #59a6ff; 
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: background-color 0.3s, border-color 0.3s;
}

.stDownloadButton > button:hover {
    background-color: #79caff;
    border-color: #79caff;
    color: #0b1220 !important;
    box-shadow: 0 0 10px rgba(89, 166, 255, 0.5);
}
/* -------------------- Sidebar Toggle Button -------------------- */
[data-testid="stHeader"] {
    background-color: #172135; 
    opacity: 1 !important; 
    z-index: 10; 
}

            
/* -------------------- Popup Chatbot -------------------- */
div.stButton > button:hover::after {
    content: "Chatbot for searching and summarizing articles";
    position: absolute;
    bottom: 70px;
    right: 0;
    width: 120px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    font-size: 14px;
    opacity: 1;
    visibility: visible;
    transition: opacity 0.3s;
    z-index: 2000;
}
div.stButton > button {
    font-size: 30px !important;
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgb(92, 92, 92);
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    transition: all 0.3s ease;
    animation: float 3s ease-in-out infinite;
    z-index: 1100;
}
@keyframes float {
    0% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0); }
}
[data-testid="stDialog"] div[role="dialog"] {
    background-color: #142047cb !important; 
    border-radius: 15px !important;
    padding: 20px !important;
}
[data-testid="stDialog"] div[role="dialog"] button {
    color: #ffffff !important;  
}

[data-testid="stDialog"] div[role="dialog"] button svg {
    fill: #ffffff !important;      
}            
</style>
""", unsafe_allow_html=True)
st.title("🚀 NASA Space Biology Knowledge Engine")
st.write("Online data extraction, targeted AI summarization, and creation of a knowledge graph to uncover scientific relationships and gaps.")

# -------------------------
# Popup Chatbot
# -------------------------


# Initialize session state
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False


# Button to open the chatbot dialog
if st.button("⏫"):
    st.session_state.show_chatbot = True

# Define the chatbot dialog
@st.dialog("Chatbot")
def chatbot_modal():
    st.components.v1.iframe("http://localhost:8502/", height=400, scrolling=True)

# Show dialog if flag is True
if st.session_state.show_chatbot:
    chatbot_modal()

# -------------------------
# Space Biology Knowledge Entities (FIXED: Comprehensive)
# -------------------------
SPACE_BIOLOGY_ENTITIES = {
    # (Space Factors/Stressors)
    "Microgravity": ["microgravity", "weightlessness", "spaceflight conditions", "simulated microgravity", "micro-g", "altered gravity", "hypogravity"],
    "Radiation": ["radiation", "HZE", "galactic cosmic ray", "GCR", "space radiation", "charged particle", "ionizing radiation", "proton", "LET", "gamma"],
    "Isolation/Confinement": ["isolation", "confinement", "social separation", "long-duration mission", "psychological stress", "stress", "crew cohesion", "habitat"],
    # (Biological Systems/Pathologies)
    "Bone Loss": ["bone loss", "osteopenia", "osteoporosis", "osteoclast", "osteoblast", "bone density", "skeletal", "femur", "tibia", "bone mineral"],
    "Muscle Atrophy": ["muscle atrophy", "muscle loss", "sarcopenia", "myotube", "soleus", "gastrocnemius", "myoblast", "skeletal muscle", "quadriceps", "extensor"],
    "Cardiovascular System": ["cardiovascular", "heart", "artery", "cardiac remodeling", "vasculature", "endothelial", "blood pressure", "orthostatic intolerance", "aorta", "atrial"],
    "Immune System": ["immune", "T-cell", "leukocyte", "cytokine", "inflammation", "immunodeficiency", "NK cell", "lymphocyte", "interleukin", "neutrophil", "macrophage"],
    "CNS/Neurological": ["central nervous system", "CNS", "brain", "hippocampus", "neuroinflammation", "cognitive decline", "neuron", "behavior", "vision", "neurovascular", "gaze"],
    "Gene/Omics": ["gene expression", "transcriptome", "proteome", "metabolome", "omics", "microRNA", "RNA-seq", "epigenetics", "DNA repair", "sequencing", "qPCR", "protein"],
    # (Models/Countermeasures)
    "Microbiome/Microbes": ["microbiome", "bacteria", "fungi", "gut microbiota", "ISS environment", "pathogen", "microbial", "biofilm", "p. aeruginosa", "stool sample"],
    "Plant Biology": ["plant", "arabidopsis", "seedling", "root growth", "crop", "plant development", "gravity sensing", "lettuce", "wheat"],
    "Artificial Gravity": ["artificial gravity", "centrifugation", "short-arm centrifuge", "re-adaptation", "rotation", "centrifuge"],
    "Countermeasure/Drug": ["exercise", "drug treatment", "countermeasure", "pharmacological", "bisphosphonate", "nutrition", "antioxidant", "supplementation", "training"],
}
ENTITY_NAMES = list(SPACE_BIOLOGY_ENTITIES.keys())

# -------------------------
# Settings & paths
# -------------------------
NCBI_API_KEY = os.getenv("05d60bc6977f33cb391be15db599e79c4e09", "") 
EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CSV_PATH = "SB_publication_PMC.csv"

# -------------------------
# Load dataset & models (cached)
# -------------------------
@st.cache_data
def load_dataset(path=CSV_PATH):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["idx"] = df.index.astype(int)
    return df

@st.cache_resource(show_spinner="Loading AI (Pegasus) Summarization Model...")
def load_summarizer_model(model_name="google/pegasus-xsum"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource(show_spinner="Loading Semantic Embedder Model...")
def load_embedder_model(model_name="all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

# CSV
if not os.path.exists(CSV_PATH):
    st.error(f"CSV dataset file not found in `{CSV_PATH}`. Please include it next to app.py.")
    st.stop()

df = load_dataset()
tokenizer, seq2seq_model = load_summarizer_model()
embed_model = load_embedder_model()

def seq2seq_summarize(text, max_length=160, min_length=50):
    if not text: return ""
    
    # ***NEW: Manage maximum input length. The PEGAGUS-XSUM model takes a maximum of 1024 tokens.
    # If the text is too long, we truncate it. (Important for hypertext summarization)
    if len(tokenizer.tokenize(text)) > 1024:
        text = tokenizer.decode(tokenizer.encode(text, truncation=True, max_length=1024))
        
    inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt", max_length=1024)
    ids = seq2seq_model.generate(**inputs, num_beams=4, max_length=max_length, min_length=min_length, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# -------------------------
# NCBI EFetch & Entity Helpers (Final)
# -------------------------
def extract_pmcid(link):
    if not isinstance(link, str): return None
    m = re.search(r"(PMC\d+)", link, re.IGNORECASE)
    return m.group(1) if m else None

@lru_cache(maxsize=2048)
def efetch_pmc_xml(pmcid: str):
    params = {"db": "pmc", "id": pmcid, "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = requests.get(EFETCH_BASE, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(0.34)
    return r.text

def parse_sections(xml_text):
    soup = BeautifulSoup(xml_text, "xml")
    
    # 1. Abstract
    abstract = None
    ab = soup.find("abstract")
    if ab:
        paras = [p.get_text(" ", strip=True) for p in ab.find_all("p")]
        abstract = " ".join(paras).strip()

    # 2. Sections (Results, Discussion, Conclusion)
    sections = {}
    body_parts = []
    for sec in soup.find_all("sec"):
        title_tag = sec.find("title")
        paras = [p.get_text(" ", strip=True) for p in sec.find_all("p")]
        content = " ".join(paras).strip()
        body_parts.append(content)
        
        if title_tag and content:
            title = title_tag.get_text(" ", strip=True)
            if "result" in title.lower() and "Results" not in sections:
                sections["Results"] = content
            elif "discussion" in title.lower() and "Discussion" not in sections:
                sections["Discussion"] = content
            elif "conclusion" in title.lower() and "Conclusion" not in sections:
                sections["Conclusion"] = content
            else:
                sections[title] = content
    
    # 3. Body Fallback
    body = " ".join(body_parts).strip()
    if not body:
        b = soup.find("body")
        if b:
            paras = [p.get_text(" ", strip=True) for p in b.find_all("p")]
            body = " ".join(paras).strip()
            
    return {"abstract": abstract, "sections": sections, "body": body}

def select_text(parsed, prefer="Abstract"):
    
    if "Abstract" in prefer and parsed.get("abstract"):
        return parsed["abstract"]
    
    if "Results" in prefer and parsed["sections"].get("Results"):
        return parsed["sections"]["Results"]
        
    if "Conclusion" in prefer and parsed["sections"].get("Conclusion"):
        return parsed["sections"]["Conclusion"]

    if "Discussion" in prefer and parsed["sections"].get("Discussion"):
        return parsed["sections"]["Discussion"]
        
    if "Body" in prefer and parsed.get("body"):
        return parsed["body"]
        
    # Fallback
    if parsed.get("abstract"): return parsed["abstract"]
    if parsed.get("body"): return parsed["body"]
    
    return None

def extract_entities(text, title):
    """**Modified:** Extract key entities based on keywords, with reference to title if no text exists."""
    
    target_text = text if text else title
    
    if not target_text:
        return []
        
    text_lower = target_text.lower()
    found_entities = set()
    for entity_name, keywords in SPACE_BIOLOGY_ENTITIES.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                found_entities.add(entity_name)
                break
    return list(found_entities)

def find_genelab_id(full_text):
    """**Modified:** Finding GeneLab/GEO/SRA/PRIDE identifiers in the full text of the article. (Comprehensive)"""
    if not full_text:
        return []
    links = []
    
    # 1. GeneLab Data System (GLDS)
    glds = re.findall(r'(GLDS[-_]?\d+)', full_text, re.IGNORECASE)
    for id_val in glds:
        formatted_id = id_val.replace('_', '-')
        if f"GLDS: {formatted_id}" not in links:
            links.append(f"GLDS: {formatted_id}")

    # 2. GEO Series (GSE) and Sample (GSM)
    gse_gsm = re.findall(r'(GSE\d+|GSM\d+)', full_text, re.IGNORECASE)
    for id_val in gse_gsm:
        prefix = 'GSE' if id_val.startswith('GSE') else 'GSM'
        output_str = f"GEO ({prefix}): {id_val}"
        if output_str not in links:
            links.append(output_str)
            
    # 3. SRA Project IDs (Sequence Read Archive - SRP/ERP/DRP)
    sra = re.findall(r'((SRP|DRP|ERP)\d+)', full_text, re.IGNORECASE)
    for match in sra:
        id_val = match[0]
        if f"SRA: {id_val}" not in links:
            links.append(f"SRA: {id_val}")
            
    # 4. ProteomeXchange/PRIDE (PXD)
    pxd = re.findall(r'(PXD\d+)', full_text, re.IGNORECASE)
    for id_val in pxd:
        if f"PRIDE: {id_val}" not in links:
            links.append(f"PRIDE: {id_val}")
        
    return links
    
# -------------------------
# TF-IDF keywords
# -------------------------
def extract_keywords(texts, top_n=10):
    vect = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vect.fit_transform([t if t else "" for t in texts])
    features = vect.get_feature_names_out()
    arr = X.toarray()
    kws_list = []
    for row in arr:
        idxs = row.argsort()[-top_n:][::-1]
        kws = [features[i] for i in idxs if row[i] > 0]
        kws_list.append(kws)
    return kws_list

# -------------------------
# Knowledge Graph (Entity-Centric)
# -------------------------
def build_entity_kg(df_subset, entity_lists):
    """Building a Knowledge Graph with article nodes and entity nodes."""
    G = nx.Graph()
    article_ids = df_subset["idx"].astype(int).tolist()
    article_titles = df_subset["Title"].tolist()

    # 1.(Article Nodes)
    for nid, title in zip(article_ids, article_titles):
        G.add_node(int(nid), 
                   label=f"P: {title[:60]}...", 
                   type="Article", 
                   color="#00d4ff", 
                   title=title,
                   size=15)
        
    # 2. (Entity Nodes)
    entity_nodes = {}
    for entity_name in ENTITY_NAMES:
        entity_node_id = f"E-{entity_name}"
        entity_nodes[entity_name] = entity_node_id
        G.add_node(entity_node_id, 
                   label=entity_name, 
                   type="Entity", 
                   color="#50ffff", 
                   size=20, 
                   title=f"Key entity: {entity_name}")
    
    # 3.Creating Edges from Article to Entity
    for i, nid in enumerate(article_ids):
        entities = entity_lists[i]
        valid_entities = [e for e in entities if e in entity_nodes] 
        
        for entity_name in valid_entities:
            entity_id = entity_nodes[entity_name]
            G.add_edge(nid, entity_id, 
                       title=f"Talk about: {entity_name}", 
                       type="DESCRIBES", 
                       color="#CCCCCC", 
                       weight=1)
            G.nodes[nid]["size"] = G.nodes[nid].get("size", 15) + 1 
            G.nodes[entity_id]["size"] = G.nodes[entity_id].get("size", 20) + 2

    # 4.Creating implicit edges between entities (co-occurrence)
    for i in range(len(article_ids)):
        entities = [e for e in entity_lists[i] if e in entity_nodes]
        for idx1 in range(len(entities)):
            for idx2 in range(idx1 + 1, len(entities)):
                e1_id = entity_nodes[entities[idx1]]
                e2_id = entity_nodes[entities[idx2]]
                
                if G.has_edge(e1_id, e2_id):
                    G[e1_id][e2_id]["weight"] += 1
                    G[e1_id][e2_id]["title"] = f"Relationship in {G[e1_id][e2_id]['weight']} Shared article"
                else:
                    G.add_edge(e1_id, e2_id, 
                               weight=1, 
                               color="#50ffff", 
                               title="Connection through 1 shared article", 
                               type="CO_OCCURRENCE",
                               dashes=True)
                    
    return G

# -------------------------
# Pyvis Render (FIXED for Unicode/Permission)
# -------------------------
def render_pyvis(G, df_subset, height="720px"):
    # Dark Mode
    net = Network(height=height, width="100%", bgcolor="#0b132b", font_color="white", cdn_resources="in_line")
    net.from_nx(G)
    
    # General Graph Settings (FIX: Use Direct Assignment)
    net.options = {
        "layout": {"improvedLayout": True},
        "physics": {
            "enabled": True, 
            "barnesHut": {
                "gravitationalConstant": -10000, 
                "centralGravity": 0.3, 
                "springLength": 95, 
                "springConstant": 0.04
            },
            "minVelocity": 0.75
        },
        "interaction": {"hover": True, "tooltipDelay": 200, "zoomView": True, "dragNodes": True},
        "edges": {"smooth": False}
    }
    
    # Final node settings
    for node in net.nodes:
        if node.get('type') == 'Article':
            article_info = df_subset[df_subset["idx"] == node["id"]]
            if not article_info.empty:
                full_title = article_info["Title"].iloc[0]
                node["title"] = full_title
                node["label"] = (full_title[:30] + "...") if len(full_title) > 30 else full_title
                node["color"] = {"border": "#00d4ff", "background": "#006699", "highlight": {"background": "#ffffff", "border": "#ffffff"}}
            else:
                node["label"] = "Uncertain article"
                
        elif node.get('type') == 'Entity':
            node["label"] = node['label'].replace(' ', '\n')
            node["color"] = {"border": "#ff7f50", "background": "#cc4d14", "highlight": {"background": "#ffffff", "border": "#ffffff"}}
            
    # Final Edge Settings
    for edge in net.edges:
        w = edge.get("weight", 1)
        if edge.get("type") == "DESCRIBES":
            edge["color"] = {"color": "#CCCCCC", "highlight": "#59a6ff"}
            edge["width"] = 0.5
        elif edge.get("type") == "CO_OCCURRENCE":
            edge["color"] = {"color": "#ff7f50", "highlight": "#ffcc00"}
            edge["width"] = 1 + float(w) * 1.5

    # Unicode/PermissionError 
    html_filename = "graph_temp.html"
    
    # 1.HTML 
    try:
        net.write_html(html_filename)
    except Exception:
        # Fallback manual write with explicit UTF-8 encoding
        with open(html_filename, "w", encoding='utf-8') as f:
             f.write(net.html)
    
    # 2. Reading file content
    html = ""
    try:
        with open(html_filename, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        st.error("Error: Temporary graph file not found.")
        return ""

    # 3. Deletes the temporary file (first allows the file to be unlocked)
    try:
        os.unlink(html_filename) 
    except Exception:
        # Ignore silent error if the file is still somehow locked by Streamlit
        pass 
    
    return html

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Dashboard settings")
max_results = st.sidebar.number_input("Maximum results to display", min_value=1, max_value=50, value=12)
top_k_keywords = st.sidebar.number_input("Number of keywords (TF-IDF)", min_value=3, max_value=20, value=10)
use_abstract_only = st.sidebar.checkbox("Use Abstract only (faster)", value=False)
section_preference = st.sidebar.selectbox("Section priority for summarization (Actionable Insight)", 
                                          ["Abstract", "Results", "Conclusion", "Discussion", "Body"])
st.sidebar.markdown("---")
st.sidebar.caption("Knowledge graph: shows the relationships between entities (e.g., from Microgravity to Bone Loss).")

st.write(f"📚 Articles in the dataset: **{len(df)}**")


# -------------------------
# Main UI flow
# -------------------------
st.subheader("🔎 Targeted search and summarization")

query = st.text_input("Enter a word or phrase to search for in the title:")
if not query:
    st.info("Enter a search term to get started.")
else:
    matches = df[df["Title"].str.contains(query, case=False, na=False)]
    df_res = matches.head(int(max_results)).copy().reset_index(drop=True)
    
    st.write(f"🔍 Results found: **{len(matches)}** (Display **{len(df_res)}** result)")
    
    if len(df_res) == 0:
        st.info("No articles were found matching the phrase.")
    else:
        df_res["idx"] = df_res["idx"].astype(int)

        texts = [] 
        entity_lists = [] 
        genelab_links = [] 
        sources = []
        
        with st.spinner("Retrieving full text and extracting entities..."):
            for _, row in df_res.iterrows():
                link = row.get("Link", "")
                pmc = extract_pmcid(link)
                parsed = {"abstract": None, "sections": {}, "body": None}
                full_text_content = ""
                
                if pmc:
                    try:
                        xml = efetch_pmc_xml(pmc)
                        parsed = parse_sections(xml)
                        # Combining abstract and body for full-text search
                        full_text_content = (parsed.get("abstract", "") + " " + parsed.get("body", "")).strip()
                        sources.append("PMC EFETCH")
                    except Exception as e:
                        sources.append(f"EFETCH failed: {str(e)[:40]}...")
                else:
                    sources.append("No PMC link")
                    
                # Entity extraction (use title if text is unavailable)
                entities = extract_entities(full_text_content, row["Title"])
                entity_lists.append(entities)
                # Raw data ID extraction (full text only)
                genelab_links.append(find_genelab_id(full_text_content))

                # Selecting text for summarization (based on user preference)
                sel = select_text(parsed, prefer=section_preference)
                if not sel:
                    sel = row["Title"]
                    sources[-1] += " | Fallback Title"

                texts.append(sel)

        # keywords & summaries
        st.subheader("Running AI models...")
        progress_bar = st.progress(0)
        
        kw_lists = extract_keywords(texts, top_n=top_k_keywords)
        summaries = []
        for i, t in enumerate(texts):
            try:
                s = seq2seq_summarize(t, max_length=160, min_length=50)
            except Exception:
                # Fallback to short sentences if summarization fails
                s = " ".join(re.split(r'(?<=[.!?])\s+', (t or ""))[:2])[:400]
            summaries.append(s)
            progress_bar.progress((i + 1) / len(texts))
        
        progress_bar.empty()


        # =================================================================
        # (Individual Summaries)
        # =================================================================
        st.markdown("### 📚 Summarized results and insights")
        for i, row in df_res.iterrows():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"**{row['Title']}**")
                if row.get("Link"):
                    st.markdown(f"[🔗 View the full article]({row.get('Link')})", unsafe_allow_html=True)
                
                st.markdown(f"**summary ({section_preference.split(' ')[0]}):**")
                st.write(summaries[i])
                
            with c2:
                st.markdown(f"**🔑 Extracted entities:**")
                entities_str = ', '.join(entity_lists[i]) if entity_lists[i] else "Not found (full text likely unavailable)"
                st.markdown(f'<div class="small">{entities_str}</div>', unsafe_allow_html=True)

                st.markdown(f"**🧬 OSDR:**")
                genelab_str = ', '.join(genelab_links[i]) if genelab_links[i] else "Not found (full article text required)"
                st.markdown(f'<div class="small">{genelab_str}</div>', unsafe_allow_html=True)


            st.markdown('</div>', unsafe_allow_html=True)


        # =================================================================
        # ***NEW: Meta-Summarization Section***
        # =================================================================
        st.markdown("---")
        st.markdown("## 🧠 Comprehensive summary of search results (knowledge integration)")
        
        if summaries:
            # 1. Combining all article summaries
            combined_text = ". ".join([s for s in summaries if s])
            
            # 2. Managing model input length (important: if summaries are too long)
            max_input_length = 1000 # token
            tokenized_combined = tokenizer.encode(combined_text, truncation=True, max_length=max_input_length)
            final_input_text = tokenizer.decode(tokenized_combined)
            
            # 3. Generating abstractive summarization (longer output)
            with st.spinner("Combining and generating comprehensive summary..."):
                final_meta_summary = seq2seq_summarize(
                    final_input_text, 
                    max_length=250, # Longer to cover multiple articles
                    min_length=80
                )
            
            st.info(f"This summary is extracted from {len(summaries)} individual article summaries:")
            st.markdown(f"**{final_meta_summary}**")
            
            if len(tokenized_combined) >= max_input_length:
                 st.caption("Note: The input length for summarization has been shortened to prevent errors. If the results are too extensive, some details may be omitted.")
        else:
            st.warning("No summaries found to combine.")
            
        st.markdown("---")
        
        # =================================================================
        # Building and rendering KG
        # =================================================================
        st.markdown("### 🔗 Knowledge Graph — Network of articles and spatial entities")
        
        G = build_entity_kg(df_res, entity_lists)
        
        if G.number_of_nodes() <= len(df_res) + 1:
            st.info("No sufficient key entities were found in these results to build a semantic knowledge graph.")
        else:
            html = render_pyvis(G, df_res)
            st.components.v1.html(html, height=720, scrolling=True)


        # =================================================================
        # Keyword analysis (Plotly)
        # =================================================================
        st.markdown("### 📊 Key entity frequency analysis")
        flat_entity = [e for sub in entity_lists for e in sub]
        
        if flat_entity:
            kw_counts = pd.Series(flat_entity).value_counts().reset_index()
            kw_counts.columns = ["Entity","Count"]
            fig = px.bar(kw_counts.head(20), x="Count", y="Entity", title="Frequency of key entities in the results", color="Entity", color_continuous_scale=px.colors.sequential.Plasma_r)
            st.plotly_chart(fig, use_container_width=True)
            
        
        # =================================================================
        # Download final DataFrame
        # =================================================================
        out_df = pd.DataFrame({
            "idx": df_res["idx"].tolist(),
            "Title": df_res["Title"].tolist(),
            "Source": sources,
            "Keywords_TFIDF": [", ".join(k) for k in kw_lists],
            "Entities_Extracted": [", ".join(e) for e in entity_lists],
            "Summary": summaries,
            "GeneLab_Links": [", ".join(g) for g in genelab_links],
            "Link": df_res["Link"].tolist()
        })
        st.download_button("⬇️ Download full insights CSV", out_df.to_csv(index=False).encode("utf-8"), file_name="nasa_knowledge_insights.csv", mime="text/csv")
        st.dataframe(out_df[["Title","Entities_Extracted","Summary","GeneLab_Links"]], use_container_width=True)



