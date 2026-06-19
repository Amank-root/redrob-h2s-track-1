"""
scoring.py — All scoring logic, fully modular and testable.

Each scorer returns a float in [0.0, 1.0].
"""

import re
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

from config import (
    CONSULTING_FIRMS, WRONG_DOMAIN_SKILLS, CORE_REQUIRED_SKILLS,
    AI_ML_TITLES, PRODUCT_ADJACENT_TITLES, LOCATION_SCORES,
    TIER1_CITIES, TIER1_INDIA, PROFICIENCY_WEIGHT, COMPANY_SIZE_SCORE,
    experience_score, notice_score, recency_score, REFERENCE_DATE,
)

REF_DATE = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Hard disqualifiers
# ════════════════════════════════════════════════════════════════════════════

def is_consulting_only(c: dict) -> bool:
    """True if ALL career history is at consulting/IT-services firms."""
    career = c.get("career_history", [])
    if not career:
        return False
    for role in career:
        company_lower = role.get("company", "").lower()
        industry_lower = role.get("industry", "").lower()
        # Check company name
        if any(cf in company_lower for cf in CONSULTING_FIRMS):
            continue
        # Check industry
        if "it services" in industry_lower or "outsourcing" in industry_lower:
            continue
        # This role is NOT consulting → candidate has product experience
        return False
    return True


def is_honeypot(c: dict) -> Tuple[bool, str]:
    """
    Detect honeypot candidates with impossible/inconsistent profiles.
    Returns (is_honeypot, reason).
    """
    p = c["profile"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    # Check 1: claimed YoE > sum of career durations by >3 years
    claimed_yoe = p.get("years_of_experience", 0)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if total_career_months > 0:
        career_yoe = total_career_months / 12
        if claimed_yoe > career_yoe + 4:
            return True, f"claimed YoE {claimed_yoe:.1f} >> career history {career_yoe:.1f} yrs"

    # Check 2: Expert skill with 0 duration months (self-reported with no use)
    expert_zero = [s["name"] for s in skills
                   if s.get("proficiency") == "expert"
                   and s.get("duration_months", 1) == 0]
    if len(expert_zero) >= 3:
        return True, f"expert proficiency with 0 months used: {expert_zero[:3]}"

    # Check 3: Impossible date overlap - role ends before it starts
    for role in career:
        start = role.get("start_date", "")
        end = role.get("end_date", "")
        if start and end:
            try:
                s = datetime.strptime(start, "%Y-%m-%d").date()
                e = datetime.strptime(end, "%Y-%m-%d").date()
                if e < s:
                    return True, f"role end date before start: {role.get('title')}"
            except ValueError:
                pass

    # Check 4: Excessive skill count with very short use (keyword stuffing)
    zero_duration_skills = sum(1 for s in skills if s.get("duration_months", 1) == 0)
    if zero_duration_skills >= 8:
        return True, f"{zero_duration_skills} skills with 0 months duration"

    # Check 5: Current title completely mismatches all career history
    current_title = p.get("current_title", "").lower()
    career_titles = [r.get("title", "").lower() for r in career]
    if career and not any(
        current_title[:6] in ct[:6] or ct[:6] in current_title[:6]
        for ct in career_titles[:3]
    ):
        # Title mismatch is common in career changes, only flag extreme cases
        pass

    return False, ""


def is_wrong_domain_primary(c: dict) -> bool:
    """
    True if candidate's primary expertise is CV/speech/robotics
    without meaningful NLP/IR experience.
    """
    skills = c.get("skills", [])
    career = c.get("career_history", [])

    skill_names = {s["name"].lower() for s in skills}
    career_text = " ".join(r.get("description", "").lower() for r in career)
    headline = c["profile"].get("headline", "").lower()

    wrong_domain_count = sum(1 for ws in WRONG_DOMAIN_SKILLS if ws in skill_names)
    has_nlp_ir = any(s in skill_names for s in {
        "nlp", "information retrieval", "text mining", "embeddings",
        "semantic search", "ranking", "retrieval", "transformers",
        "bert", "llm", "gpt", "language model"
    })

    # Wrong domain only if clearly primary AND no NLP/IR crossover
    if wrong_domain_count >= 4 and not has_nlp_ir:
        return True
    if "robotics" in headline or "computer vision" in headline:
        if not has_nlp_ir:
            return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Core fit score
# ════════════════════════════════════════════════════════════════════════════

def score_skills(c: dict) -> Tuple[float, str]:
    """
    Score skill match against JD requirements.
    Returns (score, reasoning_fragment).
    """
    skills = c.get("skills", [])
    sig = c.get("redrob_signals", {})
    assessment_scores = sig.get("skill_assessment_scores", {})

    matched_skills = []
    total_weight = 0.0
    max_possible = 0.0

    for s in skills:
        name_lower = s["name"].lower()
        if not any(req in name_lower or name_lower in req for req in CORE_REQUIRED_SKILLS):
            continue

        prof_w = PROFICIENCY_WEIGHT.get(s.get("proficiency", "beginner"), 0.25)
        dur_months = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        # Duration bonus: peaks at 24 months, saturates
        dur_bonus = min(1.0, dur_months / 24) * 0.3

        # Endorsement bonus: log scale
        import math
        end_bonus = min(0.2, math.log1p(endorsements) / math.log1p(100) * 0.2)

        # Assessment verification bonus
        assess_bonus = 0.0
        for akey, ascore in assessment_scores.items():
            if akey.lower() in name_lower or name_lower in akey.lower():
                assess_bonus = min(0.15, ascore / 100 * 0.15)
                break

        skill_score = min(1.0, prof_w + dur_bonus + end_bonus + assess_bonus)
        total_weight += skill_score
        max_possible += 1.0
        matched_skills.append(s["name"])

    if max_possible == 0:
        return 0.0, "no core AI/ML skills"

    raw = total_weight / max(max_possible, 1)
    # Soft cap: require at least 3 matched skills to get above 0.5
    if len(matched_skills) < 3:
        raw *= 0.5
    elif len(matched_skills) < 6:
        raw *= 0.75

    score = min(1.0, raw)
    top_skills = matched_skills[:4]
    return score, f"{len(matched_skills)} core skills: {', '.join(top_skills)}"


def score_experience_band(c: dict) -> Tuple[float, str]:
    """Score based on years of experience vs JD band (5-9 yrs)."""
    yoe = c["profile"].get("years_of_experience", 0)
    score = experience_score(yoe)
    return score, f"{yoe:.1f} yrs experience"


def score_location(c: dict) -> Tuple[float, str]:
    """Score location fit for Pune/Noida/NCR preference."""
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    location = p.get("location", "").lower()
    country = p.get("country", "").lower()
    willing_relocate = sig.get("willing_to_relocate", False)

    if country != "india":
        return LOCATION_SCORES["outside_india"], f"outside India ({p.get('country', '?')})"

    for city in TIER1_CITIES:
        if city in location:
            return LOCATION_SCORES["tier1_exact"], f"{p.get('location')} (Tier 1 exact)"

    for city in TIER1_INDIA:
        if city in location:
            return LOCATION_SCORES["tier1_india"], f"{p.get('location')} (major city)"

    if willing_relocate:
        return LOCATION_SCORES["tier2_india"], f"{p.get('location')}, willing to relocate"

    return LOCATION_SCORES["tier3_india"], f"{p.get('location')}, India"


def score_title_fit(c: dict) -> Tuple[float, str]:
    """
    Score current title and career trajectory fit.
    This is the anti-keyword-stuffing gate.
    """
    p = c["profile"]
    career = c.get("career_history", [])

    current_title = p.get("current_title", "").lower()

    # Check current title
    title_score = 0.0
    if any(t in current_title for t in AI_ML_TITLES):
        title_score = 1.0
        title_cat = "AI/ML title"
    elif any(t in current_title for t in PRODUCT_ADJACENT_TITLES):
        title_score = 0.65
        title_cat = "adjacent technical title"
    elif "engineer" in current_title or "scientist" in current_title:
        title_score = 0.50
        title_cat = "technical title"
    elif any(t in current_title for t in ["manager", "analyst", "designer", "writer", "sales", "hr ", "accountant"]):
        title_score = 0.10
        title_cat = "non-technical title"
    else:
        title_score = 0.30
        title_cat = "other title"

    # Career trajectory bonus: check if any past role was ML/AI
    has_ml_history = any(
        any(t in role.get("title", "").lower() for t in AI_ML_TITLES)
        for role in career
    )
    if has_ml_history and title_score < 1.0:
        title_score = min(1.0, title_score + 0.20)
        title_cat += " + ML history"

    return title_score, f"{p.get('current_title')} ({title_cat})"


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Career quality score
# ════════════════════════════════════════════════════════════════════════════

def score_career_quality(c: dict) -> Tuple[float, str]:
    """
    Score career quality: product company ratio, company size fit,
    title progression, GitHub, and description evidence of shipped systems.
    """
    career = c.get("career_history", [])
    sig = c.get("redrob_signals", {})

    if not career:
        return 0.3, "no career history"

    # 1. Product company ratio
    total_months = sum(r.get("duration_months", 0) for r in career)
    product_months = 0
    for role in career:
        company_lower = role.get("company", "").lower()
        industry_lower = role.get("industry", "").lower()
        is_cons = any(cf in company_lower for cf in CONSULTING_FIRMS)
        is_services = "it services" in industry_lower or "outsourcing" in industry_lower
        if not is_cons and not is_services:
            product_months += role.get("duration_months", 0)

    product_ratio = product_months / max(total_months, 1)

    # 2. Company size score (average, weighted by duration)
    size_scores = []
    for role in career:
        cs = role.get("company_size", "")
        sz = COMPANY_SIZE_SCORE.get(cs, 0.65)
        size_scores.append((sz, role.get("duration_months", 1)))
    weighted_size = sum(s * d for s, d in size_scores) / max(sum(d for _, d in size_scores), 1)

    # 3. GitHub activity
    github = sig.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.5   # no penalty for not having github
    elif github >= 70:
        github_score = 1.0
    elif github >= 40:
        github_score = 0.80
    elif github >= 15:
        github_score = 0.60
    else:
        github_score = 0.35

    # 4. Career description quality (shipped systems evidence)
    shipping_keywords = {
        "production", "deployed", "shipped", "users", "scale", "real-time",
        "latency", "throughput", "a/b test", "embedding", "ranking", "retrieval",
        "search", "recommendation", "pipeline", "inference", "api",
    }
    desc_text = " ".join(r.get("description", "").lower() for r in career[:3])
    keyword_hits = sum(1 for kw in shipping_keywords if kw in desc_text)
    description_score = min(1.0, keyword_hits / 8)

    # Composite
    score = (
        0.40 * product_ratio +
        0.20 * weighted_size +
        0.20 * github_score +
        0.20 * description_score
    )

    github_str = f"{github:.0f}" if github >= 0 else "N/A"
    reason = (
        f"product_co {product_ratio:.0%}, "
        f"github={github_str}, "
        f"shipping_signals={keyword_hits}"
    )
    return score, reason


# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Behavioral signal multiplier
# ════════════════════════════════════════════════════════════════════════════

def score_behavioral(c: dict) -> Tuple[float, str]:
    """
    Score candidate availability and engagement signals.
    A perfect-on-paper zombie gets multiplied down here.
    """
    sig = c.get("redrob_signals", {})

    # Recency of activity
    last_active_str = sig.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (REF_DATE - last_active).days
            rec_score = recency_score(max(0, days_ago))
        except ValueError:
            rec_score = 0.5
    else:
        rec_score = 0.5

    # Open to work flag
    otw_score = 1.0 if sig.get("open_to_work_flag", False) else 0.55

    # Recruiter response rate (critical signal)
    rr = sig.get("recruiter_response_rate", 0.5)
    response_score = rr  # already 0-1

    # Response time
    rt_hours = sig.get("avg_response_time_hours", 24)
    if rt_hours <= 4:    rt_score = 1.0
    elif rt_hours <= 12: rt_score = 0.90
    elif rt_hours <= 24: rt_score = 0.80
    elif rt_hours <= 48: rt_score = 0.70
    elif rt_hours <= 96: rt_score = 0.55
    else:                rt_score = 0.40

    # Notice period
    notice = sig.get("notice_period_days", 90)
    n_score = notice_score(notice)

    # Interview completion (shows up, doesn't ghost)
    ic = sig.get("interview_completion_rate", 0.7)
    ic_score = ic

    # Trust signals
    trust = (
        (0.4 if sig.get("verified_email", False) else 0.0) +
        (0.4 if sig.get("verified_phone", False) else 0.0) +
        (0.2 if sig.get("linkedin_connected", False) else 0.0)
    )
    trust_score = trust  # 0-1

    # Profile completeness
    completeness = sig.get("profile_completeness_score", 70) / 100

    # Weighted composite
    behavioral = (
        0.25 * rec_score +
        0.20 * otw_score +
        0.20 * response_score +
        0.10 * rt_score +
        0.10 * n_score +
        0.08 * ic_score +
        0.04 * trust_score +
        0.03 * completeness
    )

    reasons = []
    if rec_score < 0.5:
        reasons.append(f"inactive {days_ago}d ago")
    elif otw_score == 0.55:
        reasons.append("not open to work")
    if response_score < 0.3:
        reasons.append(f"low response rate {rr:.0%}")
    if n_score < 0.7:
        reasons.append(f"{notice}d notice")

    reason = f"behavioral={behavioral:.2f} (recency={rec_score:.2f}, response={rr:.2f}, notice={notice}d)"
    return behavioral, reason


# ════════════════════════════════════════════════════════════════════════════
# Reasoning builder
# ════════════════════════════════════════════════════════════════════════════

def build_reasoning(c: dict, scores: dict, embed_score: float) -> str:
    """Build a specific, non-templated reasoning string for the CSV."""
    p = c["profile"]
    sig = c.get("redrob_signals", {})

    # Lead with strongest signal
    title = p.get("current_title", "")
    yoe = p.get("years_of_experience", 0)
    location = p.get("location", "")
    company = p.get("current_company", "")

    top_skills = [s["name"] for s in c.get("skills", [])
                  if any(req in s["name"].lower() for req in CORE_REQUIRED_SKILLS)][:3]

    notice = sig.get("notice_period_days", "?")
    rr = sig.get("recruiter_response_rate", 0)
    otw = sig.get("open_to_work_flag", False)
    github = sig.get("github_activity_score", -1)
    last_active = sig.get("last_active_date", "")

    parts = [f"{title} @ {company} ({yoe:.1f} yrs, {location})"]

    if top_skills:
        parts.append(f"core skills: {', '.join(top_skills)}")

    if embed_score > 0.5:
        parts.append(f"semantic fit={embed_score:.2f}")

    avail_parts = []
    if otw:
        avail_parts.append("open to work")
    if notice <= 30:
        avail_parts.append(f"{notice}d notice")
    elif notice <= 60:
        avail_parts.append(f"{notice}d notice (buyout possible)")
    if rr >= 0.6:
        avail_parts.append(f"response rate {rr:.0%}")
    if github >= 40:
        avail_parts.append(f"GitHub={github:.0f}")
    if avail_parts:
        parts.append("; ".join(avail_parts))

    return ". ".join(parts)[:250]  # spec doesn't limit but keep clean
