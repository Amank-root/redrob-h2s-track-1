"""
Redrob Intelligent Candidate Ranker — Configuration
All tunable weights and constants in one place.
"""

# ─── Scoring layer weights (must sum to 1.0) ───────────────────────────────
WEIGHT_CORE_FIT      = 0.40
WEIGHT_CAREER_QUAL   = 0.35
WEIGHT_BEHAVIORAL    = 0.25

# ─── Reference date (hackathon dataset epoch) ──────────────────────────────
REFERENCE_DATE = "2026-06-19"

# ─── Hard-disqualifier: consulting-only firms ──────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree",
    "ibm india", "deloitte", "pwc", "kpmg", "ey ",
}

# ─── Wrong-domain penalty skills (CV/speech/robotics primary) ──────────────
WRONG_DOMAIN_SKILLS = {
    "robotics", "ros", "slam", "computer vision", "opencv", "image segmentation",
    "object detection", "yolo", "speech recognition", "tts", "asr",
    "autonomous vehicles", "lidar", "point cloud",
}

# ─── Core AI/ML skills the JD actually needs ───────────────────────────────
CORE_REQUIRED_SKILLS = {
    # Must-haves (higher weight)
    "embeddings", "sentence transformers", "vector database", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "hybrid search", "dense retrieval", "bm25",
    "information retrieval", "ranking", "reranking", "ndcg", "mrr",
    "learning to rank", "semantic search",
    # Strong nice-to-haves
    "python", "pytorch", "transformers", "hugging face", "nlp",
    "llm", "large language model", "fine-tuning", "lora", "qlora", "peft",
    "rag", "retrieval augmented generation", "langchain", "a/b testing",
    "recommendation", "search", "machine learning", "deep learning",
    "xgboost", "lightgbm", "evaluation framework",
}

# ─── Title tiers for career quality scoring ─────────────────────────────────
AI_ML_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
    "research engineer", "applied scientist", "data scientist", "mlops",
    "search engineer", "ranking engineer", "recommendation engineer",
    "senior ml", "senior ai", "staff ml", "principal ml",
}

PRODUCT_ADJACENT_TITLES = {
    "backend engineer", "software engineer", "data engineer",
    "full stack", "platform engineer", "infrastructure engineer",
    "analytics engineer", "data analyst",
}

# ─── Location scoring ───────────────────────────────────────────────────────
LOCATION_SCORES = {
    "tier1_exact":   1.00,   # Pune, Noida, Delhi NCR
    "tier1_india":   0.88,   # Mumbai, Hyderabad, Bangalore, Chennai
    "tier2_india":   0.72,   # Other India cities, willing to relocate
    "tier3_india":   0.50,   # Other India, not willing to relocate
    "outside_india": 0.35,
}

TIER1_CITIES = {"pune", "noida", "delhi", "gurgaon", "gurugram", "new delhi", "faridabad", "ghaziabad"}
TIER1_INDIA  = {"mumbai", "hyderabad", "bangalore", "bengaluru", "chennai", "kolkata"}

# ─── Experience band scoring (peaks at 6-8 yrs) ─────────────────────────────
def experience_score(yoe: float) -> float:
    if yoe < 3:    return 0.30
    if yoe < 5:    return 0.60 + 0.10 * (yoe - 3)
    if yoe <= 9:   return 1.00 - 0.02 * max(0, yoe - 8)
    if yoe <= 12:  return 0.75
    return 0.55

# ─── Notice period scoring ──────────────────────────────────────────────────
def notice_score(days: int) -> float:
    if days <= 0:   return 1.00
    if days <= 30:  return 1.00
    if days <= 60:  return 0.85
    if days <= 90:  return 0.72
    if days <= 120: return 0.60
    return 0.45

# ─── Activity recency scoring (days since last active) ─────────────────────
def recency_score(days_ago: int) -> float:
    if days_ago <= 7:    return 1.00
    if days_ago <= 30:   return 0.95
    if days_ago <= 90:   return 0.80
    if days_ago <= 180:  return 0.60
    if days_ago <= 365:  return 0.35
    return 0.15

# ─── Skill proficiency multipliers ─────────────────────────────────────────
PROFICIENCY_WEIGHT = {
    "expert":       1.00,
    "advanced":     0.80,
    "intermediate": 0.55,
    "beginner":     0.25,
}

# ─── Company size bonus for startup/scale-up fit ───────────────────────────
COMPANY_SIZE_SCORE = {
    "1-10":       0.70,
    "11-50":      0.85,
    "51-200":     1.00,
    "201-500":    1.00,
    "501-1000":   0.95,
    "1001-5000":  0.80,
    "5001-10000": 0.65,
    "10001+":     0.50,
}

# ─── Embedding model (CPU-friendly) ─────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 512

# ─── JD text sections to embed ──────────────────────────────────────────────
JD_CORE_TEXT = """
Senior AI Engineer role at Redrob AI, a Series A AI-native talent intelligence platform.
5-9 years experience in applied ML. Must have production experience with embeddings-based retrieval systems
using sentence-transformers, OpenAI embeddings, BGE, E5 or similar. Production experience with vector
databases: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python.
Hands-on evaluation frameworks for ranking: NDCG, MRR, MAP, A/B testing, offline-to-online correlation.
Has shipped at least one end-to-end ranking, search, or recommendation system to real users at scale.
Product company experience required, not consulting. Scrappy shipping attitude, not pure research.
LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank XGBoost neural. Hybrid retrieval dense sparse BM25.
Located in Pune Noida Delhi NCR Hyderabad Mumbai Bangalore. Short notice period preferred.
Active candidate, responds to recruiters, open to work.
"""
