# Redrob Intelligent Candidate Ranker

> **Hackathon Submission** — India Runs Data & AI Challenge  
> Task: Rank 100,000 candidates for a Senior AI Engineer role at Redrob  
> Output: Top 100 ranked candidates with reasoning, in under 30 seconds on CPU

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker on the full dataset
python rank.py --candidates candidates.jsonl --out submission.csv

# 3. Validate your submission
python validate_submission.py --csv submission.csv --candidates candidates.jsonl
```

**Expected output:**
```
Scored 100,000 candidates in ~22s
Disqualified: ~10,000 (10.0%)
Eligible: ~90,000
✅ Validation PASSED — submission.csv is clean and ready to submit
```

---

## File Structure

```
├── rank.py                   # Main ranking script (run this)
├── precompute.py             # Optional: pre-compute embeddings for +accuracy
├── scoring.py                # All scoring functions (modular, testable)
├── config.py                 # All weights, thresholds, constants
├── validate_submission.py    # Validate submission.csv before upload
├── app.py                    # Streamlit demo (sandbox)
├── requirements.txt
├── submission_metadata.yaml  # Fill in your team details
└── README.md
```

---

## Architecture: 5-Layer Scoring Pipeline

```
100K candidates.jsonl
        │
        ▼
┌──────────────────────────────────────────┐
│  LAYER 1 — Hard Disqualifiers            │  → ~10% removed
│  consulting-only · honeypots · wrong domain │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  LAYER 2 — Core Fit Score  (weight: 40%) │
│  skills · YoE · location · title gate   │
│  + optional embedding similarity        │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  LAYER 3 — Career Quality  (weight: 35%) │
│  product co. ratio · company size ·     │
│  GitHub activity · shipping signals     │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  LAYER 4 — Behavioral Multiplier (25%)  │
│  recency · open_to_work · response rate │
│  notice period · interview completion   │
└──────────────────────────────────────────┘
        │
        ▼
   final = 0.40 × core_fit + 0.35 × career_quality + 0.25 × behavioral
        │
        ▼
   Top 100 → submission.csv
```

---

## Why This Architecture Wins

### The keyword-stuffing problem (and how we solve it)

A naive semantic ranker (embed JD → cosine similarity) will rank a "Marketing Manager" with 
9 AI skills listed above an actual ML engineer, because the vocabulary overlap is high.

**Our solution:** Title-gated scoring. If `current_title` scores ≤ 0.15 (non-technical role),
the entire `core_fit` score is hard-capped at 0.35, regardless of skill list.

### The zombie candidate problem

A perfect-on-paper candidate who never responds to recruiters is worth zero in practice.
The **behavioral multiplier** is applied as a multiplicative term — a 5% response rate 
candidate with perfect skills still ranks below a good candidate who actually responds.

### The honeypot problem

The dataset contains ~80 honeypot candidates with impossible profiles:
- `expert` proficiency on 0 months of usage
- YoE claimed > career duration by >4 years
- Impossible date ranges

Layer 1 catches and removes all honeypots before scoring. Having >10% honeypots in 
your top-100 is an automatic disqualification per the submission spec.

---

## Layer-by-Layer Details

### Layer 1 — Hard Disqualifiers

| Check | Condition | Action |
|-------|-----------|--------|
| Consulting-only | All career at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini/HCL etc. | Disqualify |
| Honeypot: expert with 0 months | ≥3 skills marked "expert" with 0 duration | Disqualify |
| Honeypot: YoE inflation | Claimed YoE > career history + 4 years | Disqualify |
| Honeypot: impossible dates | Role end date before start date | Disqualify |
| Honeypot: keyword stuffing | ≥8 skills with 0 months duration | Disqualify |
| Wrong domain primary | CV/robotics primary, no NLP/IR crossover | Disqualify |

### Layer 2 — Core Fit (40%)

Sub-scores combined as weighted sum:

| Sub-score | Weight | Signal |
|-----------|--------|--------|
| Skill depth | 35% | Proficiency × duration_months × endorsements × assessment_score |
| Experience band | 20% | Peak at 6-8 yrs, linear fall-off outside 3-12 |
| Location | 15% | Pune/Noida/NCR=1.0, major India=0.88, India+relocate=0.72 |
| Title gate | 20% | AI/ML title=1.0, adjacent=0.65, non-technical=0.10 (caps core_fit at 0.35) |
| Embedding similarity | 10% | Cosine(JD, candidate) using all-MiniLM-L6-v2 |

**Skill scoring formula:**
```
skill_score = proficiency_weight + duration_bonus + endorsement_bonus + assessment_bonus
proficiency:  expert=1.0, advanced=0.80, intermediate=0.55, beginner=0.25
duration_bonus: min(1.0, months/24) × 0.3
endorsement_bonus: log(1+endorsements) / log(101) × 0.2
assessment_bonus: assessment_score/100 × 0.15 (if available)
```

### Layer 3 — Career Quality (35%)

| Sub-score | Weight | Signal |
|-----------|--------|--------|
| Product company ratio | 40% | Non-consulting career months / total months |
| Company size fit | 20% | 51-500 employees = 1.0 (Series A/B stage) |
| GitHub activity | 20% | github_activity_score: ≥70=1.0, ≥40=0.8, ≥15=0.6 |
| Shipping signal | 20% | Keyword hits in career descriptions: production, deployed, scale, A/B test... |

### Layer 4 — Behavioral Multiplier (25%)

| Signal | Weight | Notes |
|--------|--------|-------|
| Last active recency | 25% | ≤7 days=1.0, decay to 0.15 at >365 days |
| Open to work | 20% | 1.0 if True, 0.55 if False |
| Recruiter response rate | 20% | Direct linear scale |
| Response time | 10% | ≤4h=1.0, decay to 0.40 at >96h |
| Notice period | 10% | ≤30d=1.0, decay to 0.45 at >120d |
| Interview completion rate | 8% | Direct |
| Verified email/phone | 4% | Trust signals |
| Profile completeness | 3% | Redrob platform score |

---

## Optional: Semantic Embeddings for Higher Accuracy

```bash
# Step 1: Pre-compute embeddings once (~3 min on CPU for 100K)
python precompute.py --candidates candidates.jsonl --out cache/embeddings.npz

# Step 2: Run ranker with embeddings
python rank.py --candidates candidates.jsonl --out submission.csv --cache cache/embeddings.npz
```

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
- 384-dim, L2-normalized  
- ~80MB download, runs at ~500 cands/sec on CPU  
- Embeddings pre-normalized → cosine similarity = dot product (fast at inference)

Without embeddings (rule-based only): **~22 seconds** for 100K candidates  
With embeddings (full pipeline): **~5 minutes** total (3m precompute + 30s rank)

---

## Streamlit Demo

```bash
pip install streamlit
streamlit run app.py
```

- Upload any JSONL subset (up to 500 candidates for demo speed)
- See ranked table with per-candidate score breakdown
- Download submission.csv directly from the UI
- Score distribution chart for top-N

---

## Submission Format

`submission.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | string | e.g. `CAND_0000001` |
| `rank` | int | 1 to 100 |
| `score` | float | Non-increasing, in [0,1] |
| `reasoning` | string | Specific evidence from this candidate's profile |

Example reasoning:
```
Senior NLP Engineer @ Niramai (7.8 yrs, Indore). core skills: OpenSearch, FAISS, PEFT.
semantic fit=0.72. open to work; 15d notice; response rate 89%; GitHub=76
```

---

## Validation

```bash
python validate_submission.py --csv submission.csv --candidates candidates.jsonl
```

Checks: 100 rows · ranks 1-100 · no duplicates · valid IDs · non-increasing scores · 
non-empty reasoning · honeypot audit · reasoning uniqueness

---

## Runtime

| Mode | Time (CPU) | Hardware |
|------|-----------|---------|
| Rule-based only (full 100K) | ~22 seconds | CPU, no GPU |
| With embedding precompute | ~5 min total | CPU, no GPU |
| Precompute (once) | ~3 min | CPU |
| Rank with cached embeddings | ~30 seconds | CPU |

No external API calls. No GPU required. Runs fully offline.

---

## Configuration

All tunable weights and thresholds are in `config.py`:

```python
WEIGHT_CORE_FIT      = 0.40   # Layer 2 weight
WEIGHT_CAREER_QUAL   = 0.35   # Layer 3 weight
WEIGHT_BEHAVIORAL    = 0.25   # Layer 4 weight
```

Edit `config.py` to tune the scoring without touching pipeline logic.
