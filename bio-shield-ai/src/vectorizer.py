import numpy as np

class DNAVectorizer:
    def __init__(self):
        # We create a dictionary to map every 4-letter DNA alphabet combination
        self.word_to_idx = {}
        self.vocab_size = 0
        
    def build_vocabulary(self, tokenized_sequences):
        """
        Looks at all the 6-mer words in our dataset and gives each unique word a number.
        """
        unique_words = set()
        for seq in tokenized_sequences:
            for word in seq:
                unique_words.add(word)
                
        # Give each unique word an index number starting from 1
        # (We save 0 for padding empty spaces later!)
        for idx, word in enumerate(sorted(unique_words), start=1):
            self.word_to_idx[word] = idx
            
        self.vocab_size = len(self.word_to_idx) + 1 # +1 accounts for index 0
        print(f"Vocabulary Built! Total unique 6-mer words found: {len(self.word_to_idx)}")
        
    def transform_sequence(self, sequence_words, max_length=500):
        """
        Turns a list of words ['atg', 'cga'] into a list of numbers [14, 82]
        """
        numerical_seq = []
        for word in sequence_words:
            # If we know the word, get its number. If it's a weird character, use 0.
            numerical_seq.append(self.word_to_idx.get(word, 0))
            
        # Truncate if the sequence is too long, or pad with 0s if it's too short
        # Neural networks need all inputs to be the exact same size!
        if len(numerical_seq) > max_length:
            numerical_seq = numerical_seq[:max_length]
        else:
            numerical_seq = numerical_seq + [0] * (max_length - len(numerical_seq))
            
        return np.array(numerical_seq)

# Let's test our translator!
if __name__ == "__main__":
    fake_data = [
        ['atgcat', 'tgcatg', 'gcatgc'],
        ['atgcat', 'gcatgc', 'cccccc']
    ]
    
    vec = DNAVectorizer()
    vec.build_vocabulary(fake_data)
    
    sample_seq = ['atgcat', 'cccccc']
    numbers = vec.transform_sequence(sample_seq, max_length=5)
    print(f"\nOriginal words: {sample_seq}")
    print(f"Translated into numbers: {numbers}")