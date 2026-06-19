"""
precompute.py — Run once to generate and cache candidate embeddings.

Usage:
    python precompute.py \
        --candidates candidates.jsonl \
        --out cache/embeddings.npz \
        --batch-size 4096

Requirements:
    pip install sentence-transformers torch

GPU:
    Automatically uses CUDA if available.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def build_candidate_text(c: dict) -> str:
    """Build a single rich text representation of the candidate for embedding."""
    p = c["profile"]
    parts = []

    # Headline + summary
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    # Current role context
    parts.append(
        f"Current title: {p.get('current_title', '')}. "
        f"Industry: {p.get('current_industry', '')}. "
        f"Company size: {p.get('current_company_size', '')}."
    )

    # Career descriptions (most recent 3)
    for role in c.get("career_history", [])[:3]:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")[:400]
        industry = role.get("industry", "")
        parts.append(
            f"{title} at {company} ({industry}): {desc}"
        )

    # Skills
    skill_strs = []
    for s in c.get("skills", []):
        if s.get("proficiency") in (
            "expert",
            "advanced",
            "intermediate",
        ):
            skill_strs.append(s["name"])

    if skill_strs:
        parts.append("Skills: " + ", ".join(skill_strs[:20]))

    # Certifications
    certs = [
        cert["name"]
        for cert in c.get("certifications", [])
    ]

    if certs:
        parts.append("Certifications: " + ", ".join(certs))

    return " | ".join(p for p in parts if p.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Precompute candidate embeddings"
    )

    parser.add_argument(
        "--candidates",
        default="candidates.jsonl",
        help="Path to candidates.jsonl",
    )

    parser.add_argument(
        "--out",
        default="cache/embeddings.npz",
        help="Output .npz file path",
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Encoding batch size",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit candidates (for testing)",
    )

    args = parser.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: Missing dependencies.\n"
            "Run:\n"
            "pip install torch sentence-transformers"
        )
        sys.exit(1)

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("=" * 60)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {args.model}")

    t_model = time.time()

    model = SentenceTransformer(
        args.model,
        device=device,
    )

    # FP16 inference on GPU for higher throughput
    if device == "cuda":
        try:
            model.half()
            print("FP16 enabled")
        except Exception:
            print("FP16 unavailable, using FP32")

    print(
        f"Model loaded in "
        f"{time.time() - t_model:.1f}s"
    )

    print(f"Reading candidates from: {args.candidates}")

    t0 = time.time()

    ids = []
    texts = []

    with open(args.candidates, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break

            c = json.loads(line)

            ids.append(c["candidate_id"])
            texts.append(build_candidate_text(c))

    print(
        f"Loaded {len(ids):,} candidates "
        f"in {time.time() - t0:.1f}s"
    )

    print(
        f"Computing embeddings "
        f"(batch_size={args.batch_size})..."
    )

    t1 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - t1

    print(
        f"Encoded {len(ids):,} candidates in "
        f"{elapsed:.1f}s "
        f"({len(ids)/elapsed:,.0f} candidates/sec)"
    )

    embeddings = embeddings.astype(np.float32)

    print("Saving embeddings...")

    np.savez_compressed(
        args.out,
        ids=np.array(ids),
        embeddings=embeddings,
    )

    file_size_mb = os.path.getsize(args.out) / 1e6

    print(f"Saved -> {args.out}")
    print(f"File size: {file_size_mb:.1f} MB")

    if device == "cuda":
        print(
            f"Peak GPU memory: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )

    print(
        f"Total runtime: "
        f"{time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()