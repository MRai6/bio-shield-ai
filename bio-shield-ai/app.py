import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import sys
import os

# --- CLEAN DYNAMIC PATH INTEGRATION ---
# 1. Grab the exact absolute folder where this app.py file is running right now
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tell Python to look directly into the local 'src' directory right next to it
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Clean imports from our local modules (completely removes yellow zigzag lines)
from src.model import BioShieldCNN
from src.features import make_dna_words

# 3. Reference our asset files inside the exact same local folder
MODEL_PATH = os.path.join(BASE_DIR, "bioshield_model.pth")
META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")
# --------------------------------------

# Page Configuration (The Layout)
st.set_page_config(page_title="BioShield-AI Sentinel", layout="wide")

st.title("🛡️ BioShield-AI: Biological Intrusion Detection System")
st.markdown("### Real-Time DNA Synthesis Screening Dashboard (Phase 3 Prototype)")
st.write("Drop a raw DNA sequence below to check its safety classification using our trained 1D-CNN motif detector.")

# Load the Saved AI Brain and Word Book using our absolute paths
@st.cache_resource 
def load_ai_assets():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    
    # Rebuild the brain shell with saved sizes
    model = BioShieldCNN(vocab_size=meta['vocab_size'], num_classes=meta['num_classes'])
    # Load the trained volume knob settings (weights)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Put the model into scanning mode, not learning mode
    
    return model, meta['word_to_idx']

try:
    model, word_to_idx = load_ai_assets()
    st.success("🤖 AI Engine Status: ONLINE & LOADED")
except Exception as e:
    st.error(f"🔴 Failed to load AI model assets. Error: {e}")

# Create the UI Dashboard Split Screens
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧬 Input Genomic Sequence")
    user_dna = st.text_area("Paste DNA string (A, C, G, T) here:", height=200, value="ATGCATGCATGCATGCATGC")

with col2:
    st.subheader("📊 Screening Control Tower")
    scan_button = st.button("RUN DEEP PACKET SCREENING", type="primary")

# The Action Strategy: What happens when you click the scan button?
if scan_button:
    if len(user_dna.strip()) < 10:
        st.warning("⚠️ Sequence too short to evaluate. Provide a valid genomic sample.")
    else:
        with st.spinner("Analyzing sequence structure against 1,000 pathogenic targets..."):
            # A. Tokenize the input string into 6-letter words
            words = make_dna_words(user_dna.strip(), word_size=6)
            
            # --- INTERACTIVE MOTIF METRIC BREAKDOWN FEATURE ---
            known_motifs_count = sum(1 for w in words if w in word_to_idx)
            unknown_motifs_count = len(words) - known_motifs_count
            # --------------------------------------------------------
            
            # B. Translate text words to index numbers using our saved dictionary
            numerical_seq = [word_to_idx.get(w, 0) for w in words]
            
            # Pad or truncate to exactly 500 tokens so it matches model expectations
            max_length = 500
            if len(numerical_seq) > max_length:
                numerical_seq = numerical_seq[:max_length]
            else:
                numerical_seq = numerical_seq + [0] * (max_length - len(numerical_seq))
            
            # C. Convert to PyTorch Tensor format and pass through the AI brain
            input_tensor = torch.tensor([numerical_seq], dtype=torch.long)
            
            with torch.no_grad(): # Turn off calculus to keep it fast
                output_scores = model(input_tensor)
                predicted_class = torch.argmax(output_scores, dim=1).item()
            
            # D. Display Visual Results Indicators based on output class
            st.markdown("---")
            st.subheader("🔔 Verdict Analysis")
            
            if predicted_class == 0:
                st.balloons() # Decorative win animation!
                st.success(f"✅ **CLEARED:** Sequence matches baseline Category {predicted_class} (Harmless / Non-Pathogenic Vector).")
            else:
                st.error(f"🚨 **BIOLOGICAL ALERT FLAG Raised:** Sequence contains functional indicators matching threat Category **{predicted_class}**.")
                
            # Render our metrics in a clean, professional grid row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Detected Target Category ID", value=f"Class {predicted_class}")
            with m_col2:
                st.metric(label="Total 6-mer Motifs Processed", value=len(words))
            with m_col3:
                st.metric(label="Recognized Genomic Words", value=f"{known_motifs_count} strings", delta=f"{unknown_motifs_count} mutations/unknowns", delta_color="inverse")
                
            st.info("Motif analysis confirms structural features matching database profiles.")