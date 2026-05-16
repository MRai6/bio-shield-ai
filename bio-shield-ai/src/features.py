def make_dna_words(sequence, word_size=6):
    # This takes a long string like "ATGCATGC" 
    # and cuts it into overlapping pieces of length 'word_size'
    words = []
    for i in range(len(sequence) - word_size + 1):
        # Slice the string from current position 'i' to 'i + word_size'
        chunk = sequence[i : i + word_size]
        words.append(chunk.lower()) # We make it lowercase to stay consistent
    
    return words

# Testing our 'Scissors'
if __name__ == "__main__":
    example_dna = "ATGCATGC"
    print(f"Original: {example_dna}")
    print(f"6-letter words: {make_dna_words(example_dna, 6)}")