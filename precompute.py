"""
precompute.py — Run once to generate and cache candidate embeddings.

Usage:
    python precompute.py --candidates ./candidates.jsonl --out ./cache/embeddings.npz

Runtime: ~3-4 min on CPU for 100K candidates (MiniLM-L6-v2, batch 512).
Output:  cache/embeddings.npz  (ids array + embeddings matrix)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

def build_candidate_text(c: dict) -> str:
    """Build a single rich text representation of the candidate for embedding."""
    p = c["profile"]
    parts = []

    # Headline + summary
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    # Current role context
    parts.append(f"Current title: {p.get('current_title', '')}. "
                 f"Industry: {p.get('current_industry', '')}. "
                 f"Company size: {p.get('current_company_size', '')}.")

    # Career descriptions (most recent 3)
    for role in c.get("career_history", [])[:3]:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")[:400]  # truncate long descriptions
        industry = role.get("industry", "")
        parts.append(f"{title} at {company} ({industry}): {desc}")

    # Skills with proficiency (focus on advanced/expert)
    skill_strs = []
    for s in c.get("skills", []):
        if s.get("proficiency") in ("expert", "advanced", "intermediate"):
            skill_strs.append(s["name"])
    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs[:20]))

    # Certifications
    certs = [cert["name"] for cert in c.get("certifications", [])]
    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    return " | ".join(p for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="cache/embeddings.npz",
                        help="Output .npz file path")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit candidates (for testing)")
    args = parser.parse_args()

    # Lazy import so rank.py doesn't need to import ST if cache exists
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Reading candidates from: {args.candidates}")
    t0 = time.time()

    ids = []
    texts = []

    with open(args.candidates, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            c = json.loads(line)
            ids.append(c["candidate_id"])
            texts.append(build_candidate_text(c))

    print(f"Loaded {len(ids)} candidates in {time.time()-t0:.1f}s")

    print(f"Computing embeddings (batch_size={args.batch_size})...")
    t1 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-norm → cosine = dot product
    )

    elapsed = time.time() - t1
    print(f"Encoded {len(ids)} candidates in {elapsed:.1f}s  ({len(ids)/elapsed:.0f} cands/sec)")

    # Save
    np.savez_compressed(
        args.out,
        ids=np.array(ids),
        embeddings=embeddings.astype(np.float32),
    )
    size_mb = os.path.getsize(args.out + ".npz" if not args.out.endswith(".npz") else args.out) / 1e6
    print(f"Saved → {args.out}  ({size_mb:.1f} MB)")
    print(f"Total precompute time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
