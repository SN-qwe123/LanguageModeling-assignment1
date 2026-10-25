
from collections import Counter


def train_bpe(text: str, vocab_size: int):
    # Step 1: Initialize vocabulary with 256 byte values
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    # Step 2: Convert words into sequences of byte IDs
    words = [
        list(word.encode("utf-8"))
        for word in text.split()
    ]

    # Step 3: Repeatedly merge the most frequent pair
    while len(vocab) < vocab_size:

        # Count all adjacent token pairs
        pair_counts = Counter()

        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += 1

        if not pair_counts:
            break

        # Find the most frequent pair
        # Break ties by lexicographically greatest
        # pair of byte strings.
        best_pair = max(
            pair_counts,
            key=lambda p: (
                pair_counts[p],
                vocab[p[0]],
                vocab[p[1]],
            ),
        )

        a, b = best_pair

        # Create a new token
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]

        # Record the merge rule using byte strings
        merges.append((vocab[a], vocab[b]))

        # Replace occurrences of the pair
        new_words = []

        for word in words:
            new_word = []
            i = 0

            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == a
                    and word[i + 1] == b
                ):
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_words.append(new_word)

        words = new_words

        print(
            f"Merge {len(merges)}: "
            f"{vocab[a]!r} + {vocab[b]!r} "
            f"-> {vocab[new_id]!r}"
        )

    return vocab, merges


# Example
text = "low low low lower lower newest"

vocab, merges = train_bpe(text, vocab_size=261)

print("\nLearned merges:")
for pair in merges:
    print(pair)
