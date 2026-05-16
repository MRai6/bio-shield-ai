import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle  # A built-in Python tool to save our dictionary object
from dataset_builder import build_tokenized_dataset
from vectorizer import DNAVectorizer
from model import BioShieldCNN

def main():
    f_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_synthesis_screening_test_dataset.fasta"
    t_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_syntheisis_screening_test_dataset_metadata (1).tsv"
    
    # 1. Pull our data from Phase 2
    raw_df = build_tokenized_dataset(f_path, t_path)
    vectorizer = DNAVectorizer()
    vectorizer.build_vocabulary(raw_df['dna_words'])
    
    numerical_sequences = [vectorizer.transform_sequence(words, max_length=500) for words in raw_df['dna_words']]
    X = torch.tensor(np.array(numerical_sequences), dtype=torch.long)
    y = torch.tensor(raw_df['label'].values, dtype=torch.long)
    
    num_classes = int(torch.max(y).item()) + 1
    model = BioShieldCNN(vocab_size=vectorizer.vocab_size, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 2. Run our working training loop
    print("\n--- Training Model to Save ---")
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3 | Loss: {loss.item():.4f}")
        
    # 3. Freeze and save the assets to disk
    print("\n--- Freezing AI Assets for Web Dashboard ---")
    
    # Save the dial settings/weights of the neural network
    torch.save(model.state_dict(), "bioshield_model.pth")
    
    # Save the vocabulary word-to-number book so the dashboard can read user text
    meta_data = {
        'vocab_size': vectorizer.vocab_size,
        'num_classes': num_classes,
        'word_to_idx': vectorizer.word_to_idx
    }
    with open("model_meta.pkl", "wb") as f:
        pickle.dump(meta_data, f)
        
    print("Success! 'bioshield_model.pth' and 'model_meta.pkl' created in your workspace.")

if __name__ == "__main__":
    main()