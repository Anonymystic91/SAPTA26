import os
import argparse
import math
import string
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize_words(text, tokenizer): # token pre-processing step
    tokens = tokenizer.tokenize(text)
    merged = []
    for token in tokens:
        if token.startswith("##") and merged:
            merged[-1] += token[2:]
        else:
            merged.append(token)
    
    return [t for t in merged if all(c not in string.punctuation for c in t)]


def get_tf_prob(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {tok: cnt / total for tok, cnt in counts.items()}


def tfidf_weighted_kl(p_dist, q_dist, idf, epsilon=1e-10): # (Section 4, Eq. 1 - weighted KL)
    kl = 0.0
    for tok, p in p_dist.items():
        q = q_dist.get(tok, epsilon) or epsilon
        w = idf.get(tok, 0.0)
        kl += w * p * math.log(p / q)
    return kl


def main():
    parser = argparse.ArgumentParser(
        description="Compute TF–IDF–weighted KL divergence for a dataset"
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to CSV with columns 'original_text' and 'paraphrase_text'"
    )
    parser.add_argument(
        "--output_csv",
        default="tfidf_kl_results.csv",
        help="Where to write the per-sample KL results"
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        help="HuggingFace tokenizer model name"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-10,
        help="Smoothing constant for missing tokens and IDF denominator"
    )
    args = parser.parse_args()

    # Load dataset 
    df = pd.read_csv(args.input_csv)
    originals = df["original_text"].astype(str).tolist()
    paraphrases = df["paraphrase_text"].astype(str).tolist()
    assert len(originals) == len(paraphrases), "Mismatched lengths"

    print(f"Loaded {len(originals)} pairs from {args.input_csv}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build document frequencies and IDF
    doc_freq = defaultdict(int)
    N = len(originals)
    for text in originals:
        toks = set(tokenize_words(text, tokenizer))
        for tok in toks:
            doc_freq[tok] += 1
    idf = {
        tok: math.log(N / (dfreq + args.epsilon))
        for tok, dfreq in doc_freq.items()
    }

    # Compute per-sample KL
    kl_scores = []
    print("Calculating TF–IDF KL divergence...")
    for orig, para in tqdm(zip(originals, paraphrases), total=N):
        p_tf = get_tf_prob(tokenize_words(orig, tokenizer))
        q_tf = get_tf_prob(tokenize_words(para, tokenizer))
        kl_scores.append(tfidf_weighted_kl(p_tf, q_tf, idf, epsilon=args.epsilon))

    avg_kl = np.mean(kl_scores)
    print(f"Average TF–IDF KL divergence: {avg_kl:.4f}")

    # Save results
    results = pd.DataFrame({
        "original_text": originals,
        "anonymized_text": paraphrases,
        "kl_tfidf": kl_scores
    })

    results.loc[len(results)] = [
        "AVERAGE (all samples)", "-", avg_kl
    ]
    results.to_csv(args.output_csv, index=False)
    print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
