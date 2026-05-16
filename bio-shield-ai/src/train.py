import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # Added to clean up the tensor warning
from dataset_builder import build_tokenized_dataset
from vectorizer import DNAVectorizer
from model import BioShieldCNN

def main():
    # 1. Define our trusted data paths
    f_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_synthesis_screening_test_dataset.fasta"
    t_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_syntheisis_screening_test_dataset_metadata (1).tsv"
    
    print("--- Starting BioShield-AI Integration Pipeline ---")
    
    # 2. Assemble and Tokenize the Dataset
    raw_df = build_tokenized_dataset(f_path, t_path)
    
    # 3. Vectorize the text words into uniform numerical arrays
    vectorizer = DNAVectorizer()
    vectorizer.build_vocabulary(raw_df['dna_words'])
    
    print("\nVectorizing full dataset into numerical tensors...")
    numerical_sequences = [vectorizer.transform_sequence(words, max_length=500) for words in raw_df['dna_words']]
    
    # FIX WARNING: Convert to a single numpy array FIRST, then convert to a PyTorch tensor
    X_numpy = np.array(numerical_sequences)
    X = torch.tensor(X_numpy, dtype=torch.long)
    y = torch.tensor(raw_df['label'].values, dtype=torch.long)
    
    # --- SMART FIX FOR INDEXERROR ---
    # We find the absolute highest category number and add 1 to get the true class count
    unique_classes = torch.unique(y)
    num_classes = int(torch.max(y).item()) + 1
    print(f"Detected unique classification labels: {unique_classes.tolist()}")
    print(f"Setting total model output classes to: {num_classes}")
    # --------------------------------
    
    # 4. Instantiate our AI Brain using our dynamic class count
    print(f"\nInitializing 1D-CNN with vocab size: {vectorizer.vocab_size}...")
    model = BioShieldCNN(vocab_size=vectorizer.vocab_size, num_classes=num_classes)
    
    # 5. Define the Teacher (Loss Function) and the Coach (Optimizer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Run a mini 3-Epoch Training Loop to prove the prototype works!
    print("\n--- Initiating Model Training Loop ---")
    model.train() # Put the model in training mode
    
    for epoch in range(3):
        # Reset the coach's memory of past gradients
        optimizer.zero_grad()
        
        # Forward pass: Pass the data through the brain to get predictions
        outputs = model(X)
        
        # Calculate the mistake score (Loss) compared to the real answers
        loss = criterion(outputs, y)
        
        # Backward pass: Calculate how to fix the internal connections
        loss.backward()
        
        # Step: Adjust the internal weights to make fewer mistakes next time
        optimizer.step()
        
        print(f"Epoch {epoch+1}/3 Completed | Training Loss: {loss.item():.4f}")
        
    

if __name__ == "__main__":
    main()