# BioShield-AI Sentinel 🛡️
### Real-Time DNA Synthesis Screening & Cyber-Physical Threat Monitoring Tower

BioShield-AI Sentinel is an end-to-end biosecurity screening framework designed to function as an automated genomic firewall for DNA synthesis manufacturing facilities. The platform intercepts raw, unformatted nucleic acid orders and screens them against regulated biological threat profiles, ensuring pathogenic agents, functional toxins, or controlled select agents are flagged prior to physical synthesis.

---

## 🚀 Key Architectural Pillars

* **High-Resolution 1,000-Class Threat Mapping:** Maps input genomic data directly across a dense multi-class threat matrix derived from curated reference datasets, handling massive classification dimensions seamlessly.
* **1D-CNN Functional Motif Extractor:** Replaces simplistic keyword searches with a PyTorch 1D Convolutional Neural Network (`nn.Conv1d`). The architecture convolves over sequence embeddings, extracting localized structural markers and biological motifs to evaluate functional capabilities.
* **Fault-Tolerant K-mer Vectorization:** Implements a sliding-window tokenizer that handles noisy real-world data and structural mutations gracefully. Unknown or modified nucleotides are dynamically isolated as anomalies/mutations without interrupting runtime operations.
* **Decoupled Serialized Architecture:** Isolates heavy computational model optimization from production-line inference. Serialized parameters (`bioshield_model.pth`) and vocabulary metadata mappings (`model_meta.pkl`) are loaded asynchronously into memory using an optimized runtime cache.
* **Enterprise Diagnostic Control Tower:** Features real-time edge processing analytics including forward-pass latency benchmarking, historical threat quarantine logs, and active perimeter condition alerts wrapped in a clean, high-visibility user interface.

---

## 📁 Repository Layout & File Tree

Ensure your local directory aligns with the standard structure below to guarantee dynamic asset path resolution works seamlessly:

```text
bio-shield-ai/
├── .streamlit/
│   └── config.toml           # Global visual skin and white-background theme variables
├── data/                     # Local storage directory for raw datasets
├── src/                      # Underlying algorithmic engine components
│   ├── features.py           # Sliding-window K-mer feature tokenization layer
│   └── model.py              # PyTorch 1D-CNN Neural Network structural blueprint
├── app.py                    # Core enterprise Streamlit dashboard gateway script
├── bioshield_model.pth       # Saved neural network tensor weights (Volume Knobs)
├── model_meta.pkl            # Serialized vocabulary index dictionary (Translation Book)
└── README.md                 # System documentation abstract
