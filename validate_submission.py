"""
validate_submission.py — Validate your submission.csv before uploading.

Usage:
    python validate_submission.py --csv submission.csv --candidates candidates.jsonl

Checks:
  ✓ Exactly 100 rows
  ✓ Ranks 1-100 present exactly once
  ✓ No duplicate candidate_ids
  ✓ All candidate_ids exist in the candidates file
  ✓ Scores are non-increasing
  ✓ Scores in [0, 1]
  ✓ No empty reasoning fields
  ✓ Honeypot audit (heuristic check on top-10)
"""

import argparse
import csv
import json
import sys
from collections import Counter


def validate(csv_path: str, candidates_path: str, verbose: bool = True) -> bool:
    errors = []
    warnings = []

    def info(msg):
        if verbose:
            print(f"  ℹ  {msg}")

    def warn(msg):
        warnings.append(msg)
        if verbose:
            print(f"  ⚠  {msg}")

    def err(msg):
        errors.append(msg)
        if verbose:
            print(f"  ❌ {msg}")

    # ── Load valid candidate IDs ──────────────────────────────────────────────
    valid_ids = set()
    candidate_map = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            valid_ids.add(c["candidate_id"])
            candidate_map[c["candidate_id"]] = c
    info(f"Loaded {len(valid_ids):,} valid candidate IDs from {candidates_path}")

    # ── Load submission ───────────────────────────────────────────────────────
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"candidate_id", "rank", "score", "reasoning"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            err(f"Missing columns. Expected: {required_cols}, got: {reader.fieldnames}")
            return False
        for row in reader:
            rows.append(row)
    info(f"Loaded {len(rows)} rows from {csv_path}")

    # ── Check row count ───────────────────────────────────────────────────────
    if len(rows) != 100:
        err(f"Expected exactly 100 rows, got {len(rows)}")
    else:
        info("Row count: 100 ✓")

    # ── Check ranks 1-100 ────────────────────────────────────────────────────
    try:
        ranks = [int(r["rank"]) for r in rows]
        if sorted(ranks) != list(range(1, 101)):
            err(f"Ranks 1-100 not all present. Got: min={min(ranks)}, max={max(ranks)}, count={len(ranks)}")
        else:
            info("Ranks 1-100 all present ✓")
    except ValueError as e:
        err(f"Non-integer rank value: {e}")
        return False

    # ── Check duplicate IDs ───────────────────────────────────────────────────
    ids = [r["candidate_id"] for r in rows]
    id_counts = Counter(ids)
    dupes = [cid for cid, cnt in id_counts.items() if cnt > 1]
    if dupes:
        err(f"Duplicate candidate_ids: {dupes[:5]}")
    else:
        info("No duplicate candidate_ids ✓")

    # ── Check IDs are in the dataset ──────────────────────────────────────────
    bad_ids = [cid for cid in ids if cid not in valid_ids]
    if bad_ids:
        err(f"candidate_ids not in dataset: {bad_ids[:5]}")
    else:
        info("All candidate_ids valid ✓")

    # ── Check scores ──────────────────────────────────────────────────────────
    try:
        scores = [float(r["score"]) for r in rows]
    except ValueError as e:
        err(f"Non-numeric score: {e}")
        return False

    out_of_range = [s for s in scores if not (0 <= s <= 1)]
    if out_of_range:
        err(f"Scores out of [0,1] range: {out_of_range[:5]}")
    else:
        info("All scores in [0,1] ✓")

    # Non-increasing check
    violations = []
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1] + 1e-4:
            violations.append((i+1, scores[i-1], scores[i]))
    if violations:
        err(f"Scores not non-increasing at {len(violations)} positions. First: rank {violations[0][0]}: {violations[0][1]:.4f} → {violations[0][2]:.4f}")
    else:
        info("Scores non-increasing ✓")

    info(f"Score range: {min(scores):.4f} – {max(scores):.4f}")

    # ── Check reasoning ───────────────────────────────────────────────────────
    empty_reasoning = [i+1 for i, r in enumerate(rows) if not r.get("reasoning", "").strip()]
    if len(empty_reasoning) > 0:
        if len(empty_reasoning) > 5:
            err(f"{len(empty_reasoning)} rows with empty reasoning: {empty_reasoning[:5]}...")
        else:
            warn(f"{len(empty_reasoning)} rows with empty reasoning (ranks {empty_reasoning})")
    else:
        info("All reasoning fields populated ✓")

    # Check for templated reasoning (red flag for judges)
    reasoning_texts = [r.get("reasoning", "") for r in rows]
    unique_reasoning = len(set(reasoning_texts))
    if unique_reasoning < 90:
        warn(f"Only {unique_reasoning}/100 unique reasoning strings — may look templated to judges")
    else:
        info(f"Reasoning uniqueness: {unique_reasoning}/100 ✓")

    # ── Honeypot heuristic audit ──────────────────────────────────────────────
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                        "hcl", "tech mahindra", "mphasis", "hexaware"}

    honeypot_suspects_in_top100 = 0
    for rank_0, r in enumerate(rows[:100]):
        cid = r["candidate_id"]
        if cid not in candidate_map:
            continue
        c = candidate_map[cid]
        skills = c.get("skills", [])
        career = c.get("career_history", [])

        # Expert with 0 months
        expert_zero = sum(1 for s in skills
                          if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
        # Consulting only
        all_consulting = all(
            any(cf in r2.get("company", "").lower() for cf in consulting_firms)
            for r2 in career
        ) if career else False

        if expert_zero >= 3 or all_consulting:
            honeypot_suspects_in_top100 += 1
            if rank_0 < 20:
                warn(f"Potential honeypot in top-20: {cid} (rank {rank_0+1})")

    honeypot_pct = 100 * honeypot_suspects_in_top100 / max(len(rows), 1)
    if honeypot_pct > 10:
        err(f"~{honeypot_suspects_in_top100} honeypot suspects in top-100 ({honeypot_pct:.1f}%) — exceeds 10% disqualification threshold!")
    elif honeypot_pct > 5:
        warn(f"~{honeypot_suspects_in_top100} honeypot suspects in top-100 ({honeypot_pct:.1f}%) — review these")
    else:
        info(f"Honeypot audit: {honeypot_suspects_in_top100} suspects ({honeypot_pct:.1f}%) ✓")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if errors:
        print(f"❌ VALIDATION FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    elif warnings:
        print(f"⚠  VALIDATION PASSED WITH WARNINGS — 0 errors, {len(warnings)} warning(s)")
        print("   Submission is technically valid but review the warnings above.")
        return True
    else:
        print(f"✅ VALIDATION PASSED — submission.csv is clean and ready to submit")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate submission.csv")
    parser.add_argument("--csv", default="submission.csv")
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ok = validate(args.csv, args.candidates, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
