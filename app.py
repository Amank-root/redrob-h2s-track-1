"""
app.py — Streamlit demo for the Redrob Hackathon sandbox requirement.

Run:
    streamlit run app.py

The app accepts a JSONL file (≤ 100 candidates), runs the full ranking
pipeline, and displays the ranked output with score breakdowns.
"""

import io
import json
import os
import sys
import time
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Redrob Intelligent Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Hackathon demo — upload a JSONL sample to see ranked candidates with score breakdowns")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **5-Layer Scoring Pipeline**
    
    1. 🚫 Hard disqualifiers (consulting-only, honeypots, wrong domain)  
    2. 🎯 Core fit (skills + YoE + location + title)  
    3. 🏢 Career quality (product co. ratio + GitHub + shipping signals)  
    4. 📊 Behavioral signals (recency, response rate, notice period)  
    5. 📋 Composite + tie-break
    
    **Weights**
    - Core fit: 40%
    - Career quality: 35%  
    - Behavioral: 25%
    """)

    st.divider()
    use_embeddings = st.checkbox("Use semantic embeddings", value=False,
                                  help="Requires sentence-transformers. Slower but more accurate.")

# ── file upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload candidates.jsonl (up to 500 candidates for demo)",
    type=["jsonl", "json"],
    help="JSONL format — one candidate JSON per line"
)

# ── sample data button ────────────────────────────────────────────────────────
sample_col, run_col = st.columns([3, 1])
with sample_col:
    st.info("💡 Upload your JSONL file above, or use the sample candidates.jsonl from the hackathon bundle (first 100 lines for speed).")

with run_col:
    run_button = st.button("🚀 Run Ranking", type="primary", disabled=uploaded is None)

# ── run ranking ───────────────────────────────────────────────────────────────
if run_button and uploaded:
    from scoring import (
        is_consulting_only, is_honeypot, is_wrong_domain_primary,
        score_skills, score_experience_band, score_location, score_title_fit,
        score_career_quality, score_behavioral, build_reasoning,
    )
    from config import WEIGHT_CORE_FIT, WEIGHT_CAREER_QUAL, WEIGHT_BEHAVIORAL

    # Read candidates
    content = uploaded.read().decode("utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()][:500]

    candidates = []
    for line in lines:
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    st.write(f"Loaded **{len(candidates)}** candidates")

    # Score
    progress = st.progress(0, text="Scoring candidates...")
    results = []
    disq_results = []
    t0 = time.time()

    for i, c in enumerate(candidates):
        progress.progress((i + 1) / len(candidates), text=f"Scoring {i+1}/{len(candidates)}...")

        from rank import score_candidate
        r = score_candidate(c, None, None, None)

        if r["disqualified"]:
            disq_results.append({
                "candidate_id": r["candidate_id"],
                "name": c["profile"].get("anonymized_name", ""),
                "title": c["profile"].get("current_title", ""),
                "reason": r["disq_reason"],
            })
        else:
            results.append({
                **r,
                "name": c["profile"].get("anonymized_name", ""),
                "title": c["profile"].get("current_title", ""),
                "yoe": c["profile"].get("years_of_experience", 0),
                "location": c["profile"].get("location", ""),
                "country": c["profile"].get("country", ""),
                "open_to_work": c["redrob_signals"].get("open_to_work_flag", False),
                "response_rate": c["redrob_signals"].get("recruiter_response_rate", 0),
                "notice_days": c["redrob_signals"].get("notice_period_days", 90),
                "github": c["redrob_signals"].get("github_activity_score", -1),
                "last_active": c["redrob_signals"].get("last_active_date", ""),
            })

    progress.empty()

    results.sort(key=lambda r: (-r["final_score"], -r["behavioral"], r["candidate_id"]))
    top_n = min(100, len(results))

    elapsed = time.time() - t0
    st.success(f"✅ Scored {len(candidates)} candidates in {elapsed:.1f}s")

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total candidates", len(candidates))
    col2.metric("Eligible", len(results))
    col3.metric("Disqualified", len(disq_results))
    col4.metric("Top-N output", top_n)

    st.divider()

    # ── Top candidates table ───────────────────────────────────────────────────
    st.subheader(f"🏆 Top {top_n} Ranked Candidates")

    # Assign display scores
    max_s = results[0]["final_score"] if results else 1.0
    min_s = results[top_n-1]["final_score"] if top_n > 0 else 0.0
    score_range = max(max_s - min_s, 0.001)
    prev_score = None
    display_scores = []
    for r in results[:top_n]:
        norm = 0.200 + 0.799 * ((r["final_score"] - min_s) / score_range)
        norm = round(norm, 4)
        if prev_score is not None and norm >= prev_score:
            norm = prev_score - 0.0001
        norm = max(0.2000, round(norm, 4))
        display_scores.append(norm)
        prev_score = norm

    table_data = []
    for rank_0, (r, s) in enumerate(zip(results[:top_n], display_scores)):
        table_data.append({
            "Rank": rank_0 + 1,
            "ID": r["candidate_id"],
            "Name": r["name"],
            "Title": r["title"],
            "YoE": f"{r['yoe']:.1f}",
            "Location": r["location"],
            "Score": f"{s:.4f}",
            "Core Fit": f"{r['core_fit']:.3f}",
            "Career": f"{r['career_quality']:.3f}",
            "Behavioral": f"{r['behavioral']:.3f}",
            "OtW": "✅" if r["open_to_work"] else "❌",
            "Notice": f"{r['notice_days']}d",
            "Resp%": f"{r['response_rate']:.0%}",
            "GitHub": f"{r['github']:.0f}" if r["github"] >= 0 else "N/A",
            "Last Active": r["last_active"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Download button ────────────────────────────────────────────────────────
    import csv as csv_mod
    output_buf = io.StringIO()
    writer = csv_mod.writer(output_buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank_0, (r, s) in enumerate(zip(results[:top_n], display_scores)):
        writer.writerow([r["candidate_id"], rank_0 + 1, f"{s:.4f}", r["reasoning"]])

    st.download_button(
        "⬇️ Download submission.csv",
        data=output_buf.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )

    # ── Disqualified candidates ────────────────────────────────────────────────
    if disq_results:
        with st.expander(f"🚫 Disqualified candidates ({len(disq_results)})"):
            st.dataframe(pd.DataFrame(disq_results), use_container_width=True, hide_index=True)

    # ── Score distribution ────────────────────────────────────────────────────
    with st.expander("📊 Score distribution"):
        score_df = pd.DataFrame([{
            "Rank": i + 1,
            "Score": s,
            "Core Fit": r["core_fit"],
            "Career Quality": r["career_quality"],
            "Behavioral": r["behavioral"],
        } for i, (r, s) in enumerate(zip(results[:top_n], display_scores))])

        st.line_chart(score_df.set_index("Rank")[["Score", "Core Fit", "Career Quality", "Behavioral"]])
