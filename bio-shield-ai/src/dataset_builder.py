import pandas as pd
# We borrow the fixed tools we wrote in our other two files
from data_processor import get_clean_data
from features import make_dna_words

def build_tokenized_dataset(fasta_path, tsv_path):
    # Step 1: Use our fixed data_processor to get the clean table
    df = get_clean_data(fasta_path, tsv_path)
    
    print("Step 1: Raw DNA and Labels loaded successfully.")
    print(f"Step 2: Tokenizing {len(df)} DNA strings into 6-mer words...")
    
    # Step 2: Use the conveyor belt (.apply) to chop every DNA string into words
    # 'lambda x' represents each individual DNA string traveling down the belt
    df['dna_words'] = df['dna_string'].apply(lambda x: make_dna_words(x, word_size=6))
    
    return df

# This part runs ONLY when you press 'Play' directly on this file
if __name__ == "__main__":
    # Your local Windows file paths
    f_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_synthesis_screening_test_dataset.fasta"
    t_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_syntheisis_screening_test_dataset_metadata (1).tsv"
    
    # Run the assembly line
    full_dataset = build_tokenized_dataset(f_path, t_path)
    
    
    print(f"Total DNA Sequences Processed: {len(full_dataset)}")
    
    # Let's take a sneak peek at the first row's new column
    print("\nPreview of Tokenized Data (First 5 words of Sequence 1):")
    print(full_dataset['dna_words'].iloc[0][:5])
    
    print("\nPreview of the Master Table:")
    print(full_dataset[['label', 'dna_words']].head())