import pandas as pd           # For managing our dataset tables
from Bio import SeqIO        # For parsing the biological FASTA strings

def get_clean_data(fasta_path, tsv_path):
    # 1. Load the metadata file (the spreadsheet with the answers)
    labels_table = pd.read_csv(tsv_path, sep='\t')
    
    # --- SMART FIX FOR THE KEYERROR ---
    # We look for common column names used by NIST for answers (like 'label' or 'Classification')
    possible_columns = ['label', 'Classification', 'classification', 'Sequence_Class', 'status']
    found_column = None
    
    for col in labels_table.columns:
        # Strip away any hidden spaces and check if it matches our list
        if col.strip() in possible_columns or 'label' in col.lower() or 'class' in col.lower():
            found_column = col
            break
            
    if found_column is None:
        # If we still can't find it, we default to the very last column in the file
        found_column = labels_table.columns[-1]
    
    # We rename that mystery column to 'label' so our code never crashes again
    labels_table = labels_table.rename(columns={found_column: 'label'})
    # ----------------------------------
    
    # 2. Open an empty bucket to store our clean DNA strings
    dna_list = []
    
    # 3. Read the FASTA file line-by-line and collect the DNA letters
    for record in SeqIO.parse(fasta_path, "fasta"):
        dna_list.append(str(record.seq))
    
    # 4. Glue the DNA column and the unified 'label' column together
    master_table = pd.DataFrame({
        'dna_string': dna_list,
        'label': labels_table['label']
    })
    
    return master_table

# This part runs ONLY when you press 'Play' directly on this file
if __name__ == "__main__":
    # Your local windows file paths
    f_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_synthesis_screening_test_dataset.fasta"
    t_path = r"C:\Users\LENOVO\OneDrive\Desktop\Biosafety\bio-shield-ai\data\NIST_nucleic_acid_syntheisis_screening_test_dataset_metadata (1).tsv"
    
    print("Running Data Processor...")
    final_df = get_clean_data(f_path, t_path)
    print("\n--- Success! Table Created Perfectly ---")
    print(final_df.head())  # Prints out the first 5 rows of our clean table