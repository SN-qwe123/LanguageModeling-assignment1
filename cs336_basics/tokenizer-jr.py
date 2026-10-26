
def train_bpe(text, vocab_size):

    # Step 1: Initialize the vocabulary
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    #print(vocab)
    merges = []

    # Step 2: Convert text into byte sequences
    words = []

    for word in text.split():
        byte_data = word.encode("utf-8")
        byte_list = list(byte_data)
        words.append(byte_list)

    print(words)

    # Step 3: BPE training loop
    while len(vocab) < vocab_size:

        # Count the frequency of adjacent token pairs
        pair_counts = {}

        for word in words:
            for i in range(len(word) - 1):

                pair = (word[i], word[i + 1])

                if pair in pair_counts:
                    pair_counts[pair] += 1
                else:
                    pair_counts[pair] = 1

        # Stop if there are no pairs
        if len(pair_counts) == 0:
            break

        # Step 4: Find the most frequent pair
        best_pair = None
        highest_count = -1

        for pair in pair_counts:

            count = pair_counts[pair]

            if count > highest_count:
                highest_count = count
                best_pair = pair

            elif count == highest_count:
                # Tie-breaking rule:
                # Choose the lexicographically larger
                # pair of byte strings.
                current_bytes = (
                    vocab[pair[0]],
                    vocab[pair[1]]
                )

                best_bytes = (
                    vocab[best_pair[0]],
                    vocab[best_pair[1]]
                )

                if current_bytes > best_bytes:
                    best_pair = pair

        # Step 5: Create a new token
        a = best_pair[0]
        b = best_pair[1]

        new_id = len(vocab)

        new_token = vocab[a] + vocab[b]
        vocab[new_id] = new_token

        merges.append((vocab[a], vocab[b]))

        # Step 6: Replace the selected pair
        new_words = []

        for word in words:

            new_word = []
            i = 0

            while i < len(word):

                if i < len(word) - 1:

                    if word[i] == a and word[i + 1] == b:
                        new_word.append(new_id)
                        i += 2
                        continue

                new_word.append(word[i])
                i += 1

            new_words.append(new_word)

        words = new_words

        print("Merge", len(merges), ":",
              vocab[a], "+", vocab[b],
              "->", new_token)

    return vocab, merges


# Test the function
text = "low low low lower lower newest"

vocab, merges = train_bpe(text, 261)

print("\nFinal vocabulary size:", len(vocab))

print("\nLearned merges:")
for pair in merges:
    print(pair)
