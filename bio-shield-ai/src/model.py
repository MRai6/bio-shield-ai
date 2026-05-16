import torch
import torch.nn as nn

class BioShieldCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_classes=5):
        super(BioShieldCNN, self).__init__()
        
        # 1. The Embedding Layer: Turns integer codes into deep dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. The Convolutional Layer: Slides over the DNA words to find dangerous motifs
        # It takes 64-channel inputs and looks for patterns using a kernel size of 3
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        
        # 3. Activation & Pooling: Adds non-linear thinking and shrinks the data to key highlights
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # Grabs the absolute strongest signal found
        
        # 4. The Fully Connected Layer: The final decision maker mapping to our 5 categories
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape initially: [Batch_Size, Sequence_Length]
        
        # Pass through embedding: [Batch_Size, Sequence_Length, Embedding_Dim]
        x = self.embedding(x)
        
        # Conv1D expects shape: [Batch_Size, Embedding_Dim, Sequence_Length]
        # So we transpose the axes to match what the math expects
        x = x.transpose(1, 2)
        
        # Slide across patterns and find features
        x = self.conv1d(x)
        x = self.relu(x)
        
        # Pool down to the strongest feature map signals
        x = self.pool(x).squeeze(-1) # Shape: [Batch_Size, 128]
        
        # Final classification output score for categories 0 through 4
        output = self.fc(x)
        return output

if __name__ == "__main__":
    # Quick sanity check with fake dimensions
    # Imagine a mock vocabulary size of 1000 words
    model = BioShieldCNN(vocab_size=1000)
    
    # Simulate a batch of 2 fake encoded DNA sequences, each 500 tokens long
    fake_input = torch.randint(0, 1000, (2, 500))
    
    predictions = model(fake_input)
    print("--- Model Structural Integrity Check ---")
    print(f"Input batch shape: {fake_input.shape}")
    print(f"Output prediction shape: {predictions.shape} (2 sequences, 5 class scores each)")