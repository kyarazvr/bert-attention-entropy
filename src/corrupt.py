# src/corrupt.py
# Person B — Corruption functions
# Applies two word-order corruptions to sentences and saves three aligned output files:
#   - original.txt      (unchanged)
#   - np_shuffled.txt   (tokens shuffled within noun phrases only)
#   - full_shuffled.txt (all tokens shuffled randomly)
 
import os
import random
import spacy
 
 
SEED = 42
INPUT_PATH = "data/raw/sentences.txt"
OUTPUT_DIR = "data/corrupted"
 
# Run once to download the model if not already installed:
#   python -m spacy download en_core_web_sm
# Or it is included in requirements.txt and installed automatically.
nlp = spacy.load("en_core_web_sm")
 
 
# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------
 
def shuffle_within_nps(sentence: str) -> str:
    """
    Shuffles tokens only within each noun phrase detected by spaCy.
    Global word order is preserved — only local NP structure is broken.
 
    Example:
      Input:  "The big dog sat on the old mat"
      Output: "big The dog sat on old the mat"  (NPs shuffled, rest intact)
 
    Edge cases:
      - Single-token NPs are left unchanged (nothing to shuffle).
      - Sentences with no NPs are returned unchanged and flagged with a warning.
    """
    doc = nlp(sentence)
    tokens = [t.text for t in doc]
    chunks = list(doc.noun_chunks)
 
    if not chunks:
        print(f"  [WARNING] No noun phrases found, returning original: '{sentence}'")
        return sentence
 
    for chunk in chunks:
        start, end = chunk.start, chunk.end
        np_tokens = tokens[start:end]
        if len(np_tokens) > 1:
            random.shuffle(np_tokens)
            tokens[start:end] = np_tokens
 
    return " ".join(tokens)
 
 
def shuffle_full_sentence(sentence: str) -> str:
    """
    Shuffles all tokens in the sentence randomly.
    Destroys both local NP structure and global syntactic order.
 
    Example:
      Input:  "The big dog sat on the old mat"
      Output: "mat sat The on old dog big the"
    """
    tokens = sentence.split()
    random.shuffle(tokens)
    return " ".join(tokens)
 
 
# ---------------------------------------------------------------------------
# Apply corruptions
# ---------------------------------------------------------------------------
 
def apply_corruptions(input_path=INPUT_PATH, output_dir=OUTPUT_DIR, seed=SEED):
    """
    Reads sentences from input_path, applies both corruptions,
    and writes three aligned output files to output_dir.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
 
    with open(input_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
 
    print(f"Loaded {len(sentences)} sentences from {input_path}")
    print("Applying corruptions...")
 
    np_shuffled   = []
    full_shuffled = []
 
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"  Processing sentence {i}/{len(sentences)}...")
        np_shuffled.append(shuffle_within_nps(sentence))
        full_shuffled.append(shuffle_full_sentence(sentence))
 
    # Write all three files
    for name, data in [
        ("original",     sentences),
        ("np_shuffled",  np_shuffled),
        ("full_shuffled",full_shuffled),
    ]:
        path = os.path.join(output_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(data))
        print(f"Saved {name}.txt ({len(data)} lines)")
 
    print("\nCorruption complete.")
 
 
# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
 
def verify_alignment(output_dir=OUTPUT_DIR):
    """
    Confirms all three output files have the same number of lines
    in the same order. This must pass before handing off to Person C.
    """
    files = ["original.txt", "np_shuffled.txt", "full_shuffled.txt"]
    lengths = []
 
    print("\n--- Alignment Check ---")
    for fname in files:
        path = os.path.join(output_dir, fname)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        lengths.append(len(lines))
        print(f"  {fname}: {len(lines)} lines")
 
    assert len(set(lengths)) == 1, (
        f"[ERROR] Files are misaligned — line counts differ: {dict(zip(files, lengths))}"
    )
    print("  Alignment check PASSED.\n")
 
 
def corruption_stats(output_dir=OUTPUT_DIR):
    """
    Prints how many sentences were actually changed per condition.
    If a condition shows 0 changes, something is wrong with the corruption.
    """
    original_path = os.path.join(output_dir, "original.txt")
    with open(original_path, encoding="utf-8") as f:
        originals = f.readlines()
 
    print("--- Corruption Statistics ---")
    for condition in ["np_shuffled", "full_shuffled"]:
        path = os.path.join(output_dir, f"{condition}.txt")
        with open(path, encoding="utf-8") as f:
            corrupted = f.readlines()
        changed = sum(o != c for o, c in zip(originals, corrupted))
        pct = 100 * changed / len(originals)
        print(f"  {condition}: {changed}/{len(originals)} sentences changed ({pct:.1f}%)")
 
    print("-----------------------------\n")
 
 
def sample_comparison(output_dir=OUTPUT_DIR, n=5):
    """
    Prints n side-by-side examples across all three conditions.
    Run this manually to visually inspect that corruptions look correct.
    """
    files = {
        name: open(os.path.join(output_dir, f"{name}.txt"), encoding="utf-8").readlines()
        for name in ["original", "np_shuffled", "full_shuffled"]
    }
 
    print("--- Sample Comparison ---")
    for i in range(n):
        print(f"\n  [{i+1}]")
        for name, lines in files.items():
            print(f"    {name:<15}: {lines[i].strip()}")
    print("\n-------------------------\n")
 
 
if __name__ == "__main__":
    apply_corruptions()
    verify_alignment()
    corruption_stats()
    sample_comparison()