"""
rank.py — Main ranking script.

Usage (standard, with pre-computed embeddings):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Usage (first run, will auto-trigger precompute if cache missing):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --precompute

Usage (test with small sample):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --limit 500

Must complete in <5 min on CPU, no GPU, no API calls.
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

warnings.filterwarnings("ignore")

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    WEIGHT_CORE_FIT, WEIGHT_CAREER_QUAL, WEIGHT_BEHAVIORAL,
    JD_CORE_TEXT, EMBEDDING_MODEL,
)
from scoring import (
    is_consulting_only, is_honeypot, is_wrong_domain_primary,
    score_skills, score_experience_band, score_location, score_title_fit,
    score_career_quality, score_behavioral, build_reasoning,
)


# ════════════════════════════════════════════════════════════════════════════
# Embedding helpers
# ════════════════════════════════════════════════════════════════════════════

def load_embeddings(cache_path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """Load pre-computed embeddings from .npz cache."""
    data = np.load(cache_path)
    ids = data["ids"].tolist()
    embeddings = data["embeddings"]  # float32, shape (N, D), already L2-normalized
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    return id_to_idx, embeddings


def compute_jd_embedding(model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed the JD core text (tiny, fast)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    jd_emb = model.encode([JD_CORE_TEXT], normalize_embeddings=True, convert_to_numpy=True)
    return jd_emb[0]  # shape (D,)


# ════════════════════════════════════════════════════════════════════════════
# Per-candidate scoring
# ════════════════════════════════════════════════════════════════════════════

def score_candidate(
    c: dict,
    jd_emb: Optional[np.ndarray],
    id_to_idx: Optional[Dict[str, int]],
    embeddings: Optional[np.ndarray],
) -> dict:
    """
    Full 5-layer scoring pipeline for one candidate.
    Returns a dict with all scores and metadata.
    """
    cid = c["candidate_id"]
    result = {
        "candidate_id": cid,
        "disqualified": False,
        "disq_reason": "",
        "core_fit": 0.0,
        "career_quality": 0.0,
        "behavioral": 0.0,
        "embed_score": 0.0,
        "final_score": 0.0,
        "reasoning": "",
    }

    # ── Layer 1: Hard disqualifiers ─────────────────────────────────────────
    if is_consulting_only(c):
        result["disqualified"] = True
        result["disq_reason"] = "consulting-only career"
        return result

    hp, hp_reason = is_honeypot(c)
    if hp:
        result["disqualified"] = True
        result["disq_reason"] = f"honeypot: {hp_reason}"
        return result

    if is_wrong_domain_primary(c):
        result["disqualified"] = True
        result["disq_reason"] = "primary domain CV/robotics, no NLP/IR"
        return result

    # ── Layer 2: Core fit ────────────────────────────────────────────────────
    skill_s, skill_r   = score_skills(c)
    yoe_s,   yoe_r     = score_experience_band(c)
    loc_s,   loc_r     = score_location(c)
    title_s, title_r   = score_title_fit(c)

    # Embed score (cosine similarity, pre-normalized → dot product)
    embed_s = 0.5  # neutral default if no embeddings
    if jd_emb is not None and id_to_idx is not None and embeddings is not None:
        idx = id_to_idx.get(cid)
        if idx is not None:
            embed_s = float(np.dot(jd_emb, embeddings[idx]))
            embed_s = max(0.0, min(1.0, (embed_s + 1) / 2))  # map [-1,1]→[0,1]

    result["embed_score"] = embed_s

    # Weighted core fit — title is gated (keyword stuffer protection)
    # If title_score is very low, cap the whole core_fit score
    raw_core = (
        0.35 * skill_s +
        0.20 * yoe_s +
        0.15 * loc_s +
        0.20 * title_s +
        0.10 * embed_s
    )
    # Anti-stuffer: if title is non-technical, cap core_fit at 0.4
    if title_s <= 0.15:
        raw_core = min(raw_core, 0.35)

    result["core_fit"] = raw_core

    # ── Layer 3: Career quality ──────────────────────────────────────────────
    cq_s, cq_r = score_career_quality(c)
    result["career_quality"] = cq_s

    # ── Layer 4: Behavioral ──────────────────────────────────────────────────
    beh_s, beh_r = score_behavioral(c)
    result["behavioral"] = beh_s

    # ── Layer 5: Composite ───────────────────────────────────────────────────
    final = (
        WEIGHT_CORE_FIT    * raw_core +
        WEIGHT_CAREER_QUAL * cq_s +
        WEIGHT_BEHAVIORAL  * beh_s
    )
    result["final_score"] = round(final, 6)

    # ── Reasoning ────────────────────────────────────────────────────────────
    score_components = {
        "skill": skill_s, "yoe": yoe_s, "loc": loc_s,
        "title": title_s, "career": cq_s, "behavioral": beh_s,
    }
    result["reasoning"] = build_reasoning(c, score_components, embed_s)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_ranking(
    candidates_path: str,
    out_path: str,
    cache_path: str = "cache/embeddings.npz",
    limit: Optional[int] = None,
    team_id: str = "team_redrob",
) -> None:
    t_start = time.time()

    # ── Load embeddings (if available) ───────────────────────────────────────
    id_to_idx, embeddings, jd_emb = None, None, None

    if os.path.exists(cache_path):
        print(f"Loading embeddings from {cache_path}...")
        id_to_idx, embeddings = load_embeddings(cache_path)
        print(f"  → {len(id_to_idx)} candidate embeddings loaded")

        print("Computing JD embedding...")
        jd_emb = compute_jd_embedding()
        print(f"  → JD embedding shape: {jd_emb.shape}")
    else:
        print(f"WARNING: No embedding cache found at {cache_path}.")
        print("         Running without semantic similarity (rule-based only).")
        print("         For full accuracy, run: python precompute.py first.\n")

    # ── Read & score all candidates ──────────────────────────────────────────
    print(f"Scoring candidates from {candidates_path}...")
    results = []
    disq_count = 0
    n_read = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            c = json.loads(line)
            r = score_candidate(c, jd_emb, id_to_idx, embeddings)
            n_read += 1

            if r["disqualified"]:
                disq_count += 1
            else:
                results.append(r)

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t_start
                print(f"  {i+1:,} processed | {disq_count:,} disqualified | "
                      f"{elapsed:.1f}s elapsed | "
                      f"{(i+1)/elapsed:.0f} cands/sec")

    print(f"\nScored {n_read:,} candidates in {time.time()-t_start:.1f}s")
    print(f"Disqualified: {disq_count:,} ({100*disq_count/n_read:.1f}%)")
    print(f"Eligible: {len(results):,}")

    # ── Sort by final score (descending), tie-break by behavioral then id ─────
    results.sort(
        key=lambda r: (-r["final_score"], -r["behavioral"], r["candidate_id"])
    )

    # ── Take top 100 ─────────────────────────────────────────────────────────
    top100 = results[:100]

    if len(top100) < 100:
        print(f"WARNING: Only {len(top100)} eligible candidates (need 100).")
        print("  Filling remaining slots from disqualified pool...")
        # Re-score disqualified candidates with severe penalty as fallback
        # (ensures valid submission even if disqualification is too aggressive)
        all_ids = {r["candidate_id"] for r in results}
        filler_results = []

        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(top100) + len(filler_results) >= 100:
                    break
                c = json.loads(line)
                if c["candidate_id"] not in all_ids:
                    r = score_candidate(c, jd_emb, id_to_idx, embeddings)
                    r["final_score"] *= 0.3  # severe penalty
                    filler_results.append(r)

        filler_results.sort(key=lambda r: -r["final_score"])
        top100 = top100 + filler_results[:100 - len(top100)]

    # ── Assign scores: strictly non-increasing ───────────────────────────────
    # Normalize to [0.2, 0.999] range with smooth step-down
    n = len(top100)
    max_s = top100[0]["final_score"] if top100 else 1.0
    min_s = top100[-1]["final_score"] if top100 else 0.0
    score_range = max(max_s - min_s, 0.001)

    assigned_scores = []
    prev_score = None
    for rank_0, r in enumerate(top100):
        norm = 0.200 + 0.799 * ((r["final_score"] - min_s) / score_range)
        norm = round(norm, 4)
        # Enforce non-increasing
        if prev_score is not None and norm >= prev_score:
            norm = prev_score - 0.0001
        norm = max(0.2000, round(norm, 4))
        assigned_scores.append(norm)
        prev_score = norm

    # ── Write submission CSV ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_0, (r, s) in enumerate(zip(top100, assigned_scores)):
            writer.writerow([
                r["candidate_id"],
                rank_0 + 1,
                f"{s:.4f}",
                r["reasoning"],
            ])

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Submission written → {out_path}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"Top-1:   {top100[0]['candidate_id']}  score={assigned_scores[0]:.4f}")
    print(f"         {top100[0]['reasoning'][:100]}...")
    print(f"Top-10:  {[r['candidate_id'] for r in top100[:10]]}")
    print(f"Bottom:  {top100[-1]['candidate_id']}  score={assigned_scores[-1]:.4f}")
    print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════════════
# Validation helper (mirrors validate_submission.py logic)
# ════════════════════════════════════════════════════════════════════════════

def quick_validate(csv_path: str, candidates_path: str) -> bool:
    """Quick local validation before submission."""
    print(f"\nValidating {csv_path}...")
    errors = []

    # Load valid candidate IDs
    valid_ids = set()
    with open(candidates_path, "r") as f:
        for line in f:
            valid_ids.add(json.loads(line)["candidate_id"])

    # Load submission
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append(f"Ranks 1-100 not present exactly once")

    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate candidate_ids")

    bad_ids = [i for i in ids if i not in valid_ids]
    if bad_ids:
        errors.append(f"Invalid candidate_ids: {bad_ids[:3]}")

    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 0.0001:
            errors.append(f"Score not non-increasing at rank {i+1}: {scores[i-1]:.4f} → {scores[i]:.4f}")
            break

    empty_reasoning = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_reasoning > 5:
        errors.append(f"{empty_reasoning} rows with empty reasoning")

    if errors:
        print("❌ VALIDATION FAILED:")
        for e in errors:
            print(f"   • {e}")
        return False
    else:
        print("✅ Validation PASSED — submission looks good!")
        print(f"   100 unique candidates, ranks 1-100, scores non-increasing")
        return True


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Intelligent Candidate Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (with pre-computed embeddings):
  python rank.py --candidates candidates.jsonl --out submission.csv

  # Quick test on 500 candidates:
  python rank.py --candidates candidates.jsonl --out test_submission.csv --limit 500

  # Run precompute first, then rank:
  python precompute.py --candidates candidates.jsonl --out cache/embeddings.npz
  python rank.py --candidates candidates.jsonl --out submission.csv
        """
    )
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl (default: candidates.jsonl)")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path (default: submission.csv)")
    parser.add_argument("--cache", default="cache/embeddings.npz",
                        help="Embedding cache path (default: cache/embeddings.npz)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N candidates (for testing)")
    parser.add_argument("--team-id", default="team_redrob",
                        help="Your team ID for output filename")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after ranking")
    parser.add_argument("--precompute", action="store_true",
                        help="Run precompute step if cache missing")
    args = parser.parse_args()

    # Auto-trigger precompute if requested and cache missing
    if args.precompute and not os.path.exists(args.cache):
        print("Cache not found — running precompute step...")
        import subprocess
        result = subprocess.run([
            sys.executable, "precompute.py",
            "--candidates", args.candidates,
            "--out", args.cache,
        ], check=True)

    run_ranking(
        candidates_path=args.candidates,
        out_path=args.out,
        cache_path=args.cache,
        limit=args.limit,
        team_id=args.team_id,
    )

    if args.validate:
        quick_validate(args.out, args.candidates)


if __name__ == "__main__":
    main()
