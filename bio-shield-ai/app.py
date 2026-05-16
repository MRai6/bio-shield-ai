import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import sys
import os
import random
import time  # New import to measure inference speed latency
from datetime import datetime  # New import for timestamping logs

# --- CLEAN DYNAMIC PATH INTEGRATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from src.model import BioShieldCNN
from src.features import make_dna_words

MODEL_PATH = os.path.join(BASE_DIR, "bioshield_model.pth")
META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")
# --------------------------------------

st.set_page_config(page_title="BioShield-AI Sentinel", layout="wide")

# --- INITIALIZE COOPERATIVE SYSTEM MEMORY (SESSION STATE) ---
if "dna_input_value" not in st.session_state:
    st.session_state.dna_input_value = "ATGCATGCATGCATGCATGC"

if "quarantine_log" not in st.session_state:
    # This acts as our localized, running database history log
    st.session_state.quarantine_log = []

def click_generate_button():
    bases = ['A', 'C', 'G', 'T', 'A', 'C', 'G', 'T', 'H']
    st.session_state.dna_input_value = "".join(random.choice(bases) for _ in range(150))
# -------------------------------------------------------------

# Main Dashboard Title Banner
st.title("🛡️ BioShield-AI: Biological Intrusion Detection System")
st.markdown("### Cyber-Physical Threat Monitoring Tower (Phase 3 Enterprise Edition)")

@st.cache_resource 
def load_ai_assets():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    model = BioShieldCNN(vocab_size=meta['vocab_size'], num_classes=meta['num_classes'])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, meta['word_to_idx']

try:
    model, word_to_idx = load_ai_assets()
except Exception as e:
    st.error(f"🔴 AI Assets Offline: {e}")

# Sidebar Status Control Center (Feature 3: Border Radar)
with st.sidebar:
    st.header("⚡ System Perimeter Status")
    if len(st.session_state.quarantine_log) > 0 and st.session_state.quarantine_log[-1]["Verdict"] != "CLEARED":
        st.error("🚨 CONDITION: CRITICAL\nThreat Intrusion Intercepted.")
    else:
        st.success("🟢 CONDITION: NOMINAL\nPerimeter Monitoring Secure.")
        
    st.markdown("---")
    st.subheader("📡 Node Telemetry")
    st.caption("Host Machine: Lenovo LOQ")
    st.caption("Core Acceleration: PyTorch CPUTensor")
    st.caption("Database Protocol: NIST-1000 Threat Map")

# Layout Split Screens
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🧬 Input Genomic Sequence")
    st.button("🎲 Generate Sample DNA String", on_click=click_generate_button)
    user_dna = st.text_area("Paste DNA string (A, C, G, T) here:", height=180, key="dna_input_value")

with col2:
    st.subheader("📊 Screening Control Tower")
    st.write("Trigger full deep-packet algorithmic analysis on current genomic load.")
    scan_button = st.button("RUN DEEP PACKET SCREENING", type="primary")

if scan_button:
    if len(user_dna.strip()) < 10:
        st.warning("⚠️ Sequence too short to evaluate.")
    else:
        with st.spinner("Executing sequence convolving filters..."):
            
            # --- START LATENCY BENCHMARKING (Feature 2) ---
            start_time = time.time()
            
            words = make_dna_words(user_dna.strip(), word_size=6)
            known_motifs_count = sum(1 for w in words if w in word_to_idx)
            unknown_motifs_count = len(words) - known_motifs_count
            numerical_seq = [word_to_idx.get(w, 0) for w in words]
            
            max_length = 500
            if len(numerical_seq) > max_length:
                numerical_seq = numerical_seq[:max_length]
            else:
                numerical_seq = numerical_seq + [0] * (max_length - len(numerical_seq))
            
            input_tensor = torch.tensor([numerical_seq], dtype=torch.long)
            
            with torch.no_grad():
                output_scores = model(input_tensor)
                predicted_class = torch.argmax(output_scores, dim=1).item()
                probabilities = torch.nn.functional.softmax(output_scores, dim=1).flatten().numpy()
            
            # --- END LATENCY BENCHMARKING ---
            inference_latency = (time.time() - start_time) * 1000 # Convert to milliseconds
            throughput_speed = (len(user_dna) / 1000) / (inference_latency / 1000) # kbps simulation
            
            # --- SAVE INSTANCE TO QUARANTINE LOG (Feature 1) ---
            verdict_status = "CLEARED" if predicted_class == 0 else "FLAGGED THREAT"
            log_entry = {
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Packet ID": f"PKT-{random.randint(10000, 99999)}",
                "Sequence Length": len(user_dna),
                "Verdict": verdict_status,
                "Target Profile": f"Class {predicted_class}"
            }
            st.session_state.quarantine_log.append(log_entry)
            # ---------------------------------------------------

            st.markdown("---")
            st.subheader("🔔 Real-Time Verdict Analysis")
            
            if predicted_class == 0:
                st.balloons()
                st.success(f"✅ **CLEARED:** Sequence matches baseline Category {predicted_class} (Harmless Vector).")
            else:
                st.error(f"🚨 **BIOLOGICAL ALERT FLAG Raised:** Functional indicators match threat Category **{predicted_class}**.")
                
            # Render Core Metrics & Performance Diagnostics Together
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(label="Detected Profile ID", value=f"Class {predicted_class}")
            with m_col2:
                st.metric(label="Total Motifs Parsed", value=len(words))
            with m_col3:
                # Dynamic performance metrics display
                st.metric(label="Neural Network Latency", value=f"{inference_latency:.2f} ms")
            with m_col4:
                st.metric(label="Screening Throughput", value=f"{throughput_speed:.1f} Kb/s")
                
            # Charts Block
            st.markdown("### 📊 Interactive Analysis Metrics")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.write("**Sequence Composition Map**")
                composition_data = pd.DataFrame({
                    "Metric Type": ["Recognized Motifs", "Mutations/Unknowns"],
                    "Count": [known_motifs_count, unknown_motifs_count]
                })
                st.bar_chart(data=composition_data, x="Metric Type", y="Count", color="#FF4B4B" if predicted_class != 0 else "#29B6F6", horizontal=True)
                
            with chart_col2:
                st.write("**Model Confidence Spectrum**")
                top_classes = np.argsort(probabilities)[-5:][::-1]
                top_probs = probabilities[top_classes]
                confidence_data = pd.DataFrame({
                    "Target ID": [f"Class {c}" for c in top_classes],
                    "Confidence Score": [float(p) for p in top_probs]
                })
                st.bar_chart(data=confidence_data, x="Target ID", y="Confidence Score", color="#1E88E5")
            
            # Deep Packet Inspector Dropdown
            with st.expander("🔎 Deep Packet Inspector Fragment Matrix"):
                tokens_dataframe = pd.DataFrame({
                    "Fragment Index": range(1, len(words) + 1),
                    "Extracted 6-mer Token": words,
                    "Database Registry Status": ["Registered Match" if w in word_to_idx else "Unknown/Anomaly" for w in words],
                    "Assigned Vector Key": [word_to_idx.get(w, 0) for w in words]
                })
                st.dataframe(tokens_dataframe, use_container_width=True, hide_index=True)

# --- FEATURE 1 DISPLAY: SYSTEM QUARANTINE HISTORY AUDIT TRAIL ---
st.markdown("---")
st.subheader("📋 Autonomous System Audit Trail & Quarantine Log")
if len(st.session_state.quarantine_log) == 0:
    st.info("No packets processed in this session yet. Monitoring network pipeline...")
else:
    log_df = pd.DataFrame(st.session_state.quarantine_log)
    # Reverse it so the newest scans appear right at the top of the table
    st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True)