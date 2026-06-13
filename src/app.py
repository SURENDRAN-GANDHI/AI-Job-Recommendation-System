"""
AI Job Recommendation System — Streamlit Dashboard

Features

1. Upload Resume PDF
2. Extract Skills
3. Match Jobs
4. Show Match Percentage
5. Show Missing Skills
6. Show Top 5 Recommended Jobs
7. Interactive Dashboard
"""

from __future__ import annotations

import io
import sys
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# path setup 
_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR / "src"
for _p in [str(_APP_DIR), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# PDF extraction
try:
    import pdfplumber
    _PDF_BACKEND = "pdfplumber"
except ImportError:
    try:
        from pypdf import PdfReader
        _PDF_BACKEND = "pypdf"
    except ImportError:
        _PDF_BACKEND = "none"

# project imports
try:
    from skill_extractor import SkillExtractor
    from model import JobMatchingModel
    _IMPORTS_OK = True
except ImportError as _e:
    _IMPORTS_OK = False
    _IMPORT_ERR = str(_e)

logging.basicConfig(level=logging.WARNING)


# PAGE CONFIG 

st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DESIGN TOKENS  
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0F1B2D;
}
.stApp { background: #F0F5FF; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2447 0%, #19376D 60%, #1565C0 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #E8F0FE !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #FFFFFF !important; }

.hero {
    background: linear-gradient(135deg, #0B2447 0%, #1565C0 60%, #42A5F5 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 8px 32px rgba(11,36,71,0.18);
}
.hero-icon { font-size: 3.2rem; line-height: 1; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #90CAF9;
    margin: 0;
    font-weight: 400;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1565C0;
    margin: 0 0 0.6rem;
}

.card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(11,36,71,0.07);
    border: 1px solid #DDEEFF;
    height: 100%;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5C7BA8;
    margin: 0 0 0.5rem;
}
.card-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #0B2447;
    line-height: 1;
    margin: 0 0 4px;
}
.card-sub { font-size: 0.8rem; color: #7A9CC4; margin: 0; }

.match-bar-wrap {
    background: #E8F0FE;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin: 8px 0 4px;
}
.match-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1565C0, #42A5F5);
    transition: width 0.6s ease;
}

.job-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.2rem;
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #DDEEFF;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 6px rgba(11,36,71,0.05);
}
.rank-badge {
    min-width: 32px; height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, #0B2447, #1565C0);
    color: #FFFFFF;
    font-size: 0.8rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.job-info { flex: 1; min-width: 0; }
.job-title-text {
    font-size: 0.92rem; font-weight: 600; color: #0B2447;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.job-cat-text { font-size: 0.75rem; color: #7A9CC4; margin-top: 2px; }
.score-pill {
    background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
    color: #0D47A1; font-size: 0.8rem; font-weight: 700;
    padding: 4px 12px; border-radius: 99px; flex-shrink: 0;
}

.skills-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.skill-pill {
    font-size: 0.72rem; font-weight: 500;
    padding: 3px 10px; border-radius: 99px; white-space: nowrap;
}
.pill-green { background: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7; }
.pill-red   { background: #FFEBEE; color: #B71C1C; border: 1px solid #FFCDD2; }
.pill-blue  { background: #E3F2FD; color: #0D47A1; border: 1px solid #90CAF9; }
.pill-grey  { background: #F5F5F5; color: #37474F; border: 1px solid #CFD8DC; }

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #90CAF9;
    border-radius: 14px;
    padding: 1rem;
}

div[data-testid="stHorizontalBlock"] { gap: 1rem; }
.stButton > button {
    background: linear-gradient(135deg, #0B2447, #1565C0) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }
hr { border: none; border-top: 1px solid #DDEEFF; margin: 1.2rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



# HELPERS

def _extract_pdf_text(file_bytes: bytes) -> str:
    if _PDF_BACKEND == "pdfplumber":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif _PDF_BACKEND == "pypdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    return ""


@st.cache_resource(show_spinner=False)
def _load_jobs(jobs_path: str) -> pd.DataFrame:
    return pd.read_csv(jobs_path)


@st.cache_resource(show_spinner=False)
def _build_extractor() -> "SkillExtractor | None":
    if not _IMPORTS_OK:
        return None
    return SkillExtractor()


def _parse_skill_str(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _pct_colour(pct: float) -> str:
    if pct >= 70: return "#1B5E20"
    if pct >= 40: return "#E65100"
    return "#B71C1C"


def _skill_pills(skills: list[str], cls: str) -> str:
    if not skills:
        return "<span style='color:#999;font-size:0.78rem'>None</span>"
    return "".join(f'<span class="skill-pill {cls}">{s}</span>' for s in skills)


def _job_row_html(rank: int, title: str, category: str, score: float) -> str:
    pct = round(score * 100, 1)
    return f"""
    <div class="job-row">
        <div class="rank-badge">#{rank}</div>
        <div class="job-info">
            <div class="job-title-text">{title}</div>
            <div class="job-cat-text">{category}</div>
        </div>
        <div class="score-pill">{pct}%</div>
    </div>"""


def _match_bar(pct: float) -> str:
    colour = _pct_colour(pct)
    return f"""
    <div class="match-bar-wrap">
        <div class="match-bar-fill" style="width:{min(pct,100)}%;
             background:linear-gradient(90deg,{colour},{colour}88);">
        </div>
    </div>
    <p style="font-size:0.75rem;color:{colour};font-weight:600;margin:0">
        {pct:.1f}% skill match
    </p>"""


def _simple_gap(resume: list[str], job: list[str]) -> dict:
    rs, js = set(s.lower() for s in resume), set(s.lower() for s in job)
    matched = [s for s in resume if s.lower() in js]
    missing = [s for s in job if s.lower() not in rs]
    extra   = [s for s in resume if s.lower() not in js]
    pct     = round(len(matched) / max(len(job), 1) * 100, 1)
    return {"matched": matched, "missing": missing, "extra": extra, "match_pct": pct}



# SIDEBAR

with st.sidebar:
    st.markdown("## 💼 AI Job Recommender")
    st.markdown("---")
    st.markdown("### 📂 Data Paths")
    jobs_path = st.text_input("jobs.csv path", value="dataset/jobs.csv")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    vectorizer = st.selectbox("Vectorizer", ["mlb", "tfidf", "hybrid"], index=0)
    top_k = st.slider("Top N recommendations", 1, 10, 5)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Upload a resume PDF, extract your skills automatically, "
        "and get matched to the best-fit jobs with a full skill gap report."
    )
    if not _IMPORTS_OK:
        st.error(f"⚠️ Import error: {_IMPORT_ERR}")
    if _PDF_BACKEND == "none":
        st.warning("No PDF library found. Install `pdfplumber` or `pypdf`.")


# HERO

st.markdown("""
<div class="hero">
    <div class="hero-icon">💼</div>
    <div>
        <p class="hero-title">AI Job Recommendation System</p>
        <p class="hero-sub">Upload your resume → extract skills → find your best-fit roles</p>
    </div>
</div>""", unsafe_allow_html=True)



# STEP 1 — UPLOAD

st.markdown('<p class="section-label">Step 1 — Upload your resume</p>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop your resume PDF here", type=["pdf"], label_visibility="collapsed")

resume_text: str = ""
if uploaded is not None:
    raw_bytes = uploaded.read()
    resume_text = _extract_pdf_text(raw_bytes)
    if resume_text.strip():
        st.success(f"✅ PDF parsed — {len(resume_text):,} characters extracted.")
    else:
        st.error("Could not extract text. Try a text-based (not scanned) PDF.")



# STEP 2 — EXTRACT SKILLS

resume_skills: list[str] = []

if resume_text.strip():
    st.markdown("---")
    st.markdown('<p class="section-label">Step 2 — Extracted skills</p>', unsafe_allow_html=True)

    if _IMPORTS_OK:
        extractor = _build_extractor()
        with st.spinner("Extracting skills …"):
            resume_skills = extractor.extract_skill_names(resume_text)
    else:
        _COMMON = [
            "Python","Java","JavaScript","SQL","R","Go","C++","C#","Scala","Ruby",
            "React","Angular","Vue","Node.js","Django","Flask","FastAPI","Spring Boot",
            "AWS","Azure","GCP","Docker","Kubernetes","Terraform","Jenkins","CI/CD",
            "MySQL","PostgreSQL","MongoDB","Redis","Cassandra","Elasticsearch","DynamoDB",
            "Machine Learning","Deep Learning","NLP","TensorFlow","PyTorch","Scikit-learn",
            "Pandas","NumPy","Matplotlib","Power BI","Tableau","Excel",
            "Hadoop","Spark","Kafka","Airflow","Snowflake","dbt","ETL",
            "Git","GitHub","Jira","Agile","Scrum","REST API","GraphQL",
            "Photoshop","Figma","AutoCAD","SolidWorks","MATLAB",
            "Financial Analysis","SAP","Tally","Auditing","Compliance","GAAP",
            "Communication","Leadership","Teamwork","Problem Solving","Management",
        ]
        resume_skills = [
            s for s in _COMMON
            if re.search(rf"\b{re.escape(s)}\b", resume_text, re.IGNORECASE)
        ]

    col_skills, col_stats = st.columns([3, 1])
    with col_skills:
        st.markdown(
            f'<div class="card"><p class="card-title">Skills found in your resume</p>'
            f'<div class="skills-wrap">{_skill_pills(resume_skills, "pill-blue")}</div></div>',
            unsafe_allow_html=True,
        )
    with col_stats:
        st.markdown(
            f'<div class="card"><p class="card-title">Total skills</p>'
            f'<p class="card-value">{len(resume_skills)}</p>'
            f'<p class="card-sub">detected automatically</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander("✏️ Edit skills manually"):
        manual = st.text_area("Comma-separated skills", value=", ".join(resume_skills), height=100)
        if st.button("Apply edits"):
            resume_skills = [s.strip() for s in manual.split(",") if s.strip()]
            st.success(f"Updated — {len(resume_skills)} skills saved.")



# STEP 3 — MATCH

if resume_skills:
    st.markdown("---")
    st.markdown('<p class="section-label">Step 3 — Match against job listings</p>', unsafe_allow_html=True)

    jobs_df: pd.DataFrame = pd.DataFrame()
    try:
        jobs_df = _load_jobs(jobs_path)
        st.caption(f"📋 {len(jobs_df)} jobs loaded from `{jobs_path}`")
    except FileNotFoundError:
        st.error(f"jobs.csv not found at `{jobs_path}`. Update the path in the sidebar.")

    if not jobs_df.empty:
        job_skills_list = [_parse_skill_str(row) for row in jobs_df["skills"]]

        if _IMPORTS_OK:
            with st.spinner("Training matching model …"):
                model = JobMatchingModel(vectorizer_type=vectorizer)
                model.fit([resume_skills], job_skills_list)
                resume_vec = model.transform([resume_skills])
                job_vecs   = model.transform(job_skills_list)
                scores_row = model.match(resume_vec, job_vecs)[0]
        else:
            all_skills = sorted(set(resume_skills) | {s for jsl in job_skills_list for s in jsl})
            idx_map = {s: i for i, s in enumerate(all_skills)}
            def _vec(skills: list[str]) -> np.ndarray:
                v = np.zeros(len(all_skills))
                for s in skills:
                    if s in idx_map: v[idx_map[s]] = 1.0
                return v
            rv = _vec(resume_skills)
            rv_norm = np.linalg.norm(rv)
            scores_row = np.array([
                float(np.dot(rv, _vec(jsl)) / (rv_norm * np.linalg.norm(_vec(jsl)) + 1e-9))
                for jsl in job_skills_list
            ])

        top_indices = np.argsort(scores_row)[::-1][:top_k]
        best_idx    = int(top_indices[0])
        best_score  = float(scores_row[best_idx])
        best_title  = str(jobs_df["title"].iloc[best_idx])
        best_gap    = (_simple_gap(resume_skills, job_skills_list[best_idx])
                       if not _IMPORTS_OK
                       else model.skill_gap(resume_skills, job_skills_list[best_idx]))

        # KPI strip 
        st.markdown("---")
        st.markdown('<p class="section-label">At a glance</p>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f'<div class="card"><p class="card-title">Best match</p>'
                f'<p class="card-value">{round(best_score*100,1)}%</p>'
                f'<p class="card-sub">{best_title}</p></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="card"><p class="card-title">Your skills</p>'
                f'<p class="card-value">{len(resume_skills)}</p>'
                f'<p class="card-sub">detected in resume</p></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="card"><p class="card-title">Skills matched</p>'
                f'<p class="card-value">{len(best_gap["matched"])}</p>'
                f'<p class="card-sub">vs top job</p></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="card"><p class="card-title">Skills to learn</p>'
                f'<p class="card-value">{len(best_gap["missing"])}</p>'
                f'<p class="card-sub">for top job</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # Two-column: job list + skill gap 
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown(f'<p class="section-label">Top {top_k} recommended jobs</p>', unsafe_allow_html=True)
            for rank, j in enumerate(top_indices, start=1):
                cat = str(jobs_df["category"].iloc[j]) if "category" in jobs_df.columns else ""
                st.markdown(
                    _job_row_html(rank, str(jobs_df["title"].iloc[j]), cat, float(scores_row[j])),
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown('<p class="section-label">Skill gap analysis</p>', unsafe_allow_html=True)
            job_options = {
                f"#{i+1} — {jobs_df['title'].iloc[j]}": j
                for i, j in enumerate(top_indices)
            }
            selected_label = st.selectbox("Analyse gap for", list(job_options.keys()), label_visibility="collapsed")
            sel_j = job_options[selected_label]
            gap   = (_simple_gap(resume_skills, job_skills_list[sel_j])
                     if not _IMPORTS_OK
                     else model.skill_gap(resume_skills, job_skills_list[sel_j]))

            st.markdown(_match_bar(gap["match_pct"]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f'<p class="section-label" style="color:#2E7D32">✅ Matched ({len(gap["matched"])})</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="skills-wrap">{_skill_pills(gap["matched"], "pill-green")}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f'<p class="section-label" style="color:#C62828">⚠️ Missing ({len(gap["missing"])})</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="skills-wrap">{_skill_pills(gap["missing"], "pill-red")}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<p class="section-label" style="color:#37474F">➕ Extra skills you have</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="skills-wrap">{_skill_pills(gap["extra"][:15], "pill-grey")}</div>', unsafe_allow_html=True)

        # Score distribution chart
        st.markdown("---")
        st.markdown('<p class="section-label">Match score distribution across all jobs</p>', unsafe_allow_html=True)
        chart_df = (
            pd.DataFrame({"Job": jobs_df["title"].tolist(), "Score": (scores_row * 100).round(1).tolist()})
            .sort_values("Score", ascending=False)
            .head(20)
        )
        st.bar_chart(chart_df.set_index("Job")["Score"], height=300, color="#1565C0")

        # Download 
        st.markdown("---")
        st.markdown('<p class="section-label">Export results</p>', unsafe_allow_html=True)
        export_rows = []
        for rank, j in enumerate(top_indices, start=1):
            g = (_simple_gap(resume_skills, job_skills_list[j])
                 if not _IMPORTS_OK
                 else model.skill_gap(resume_skills, job_skills_list[j]))
            export_rows.append({
                "rank"     : rank,
                "job_title": jobs_df["title"].iloc[j],
                "category" : jobs_df["category"].iloc[j] if "category" in jobs_df.columns else "",
                "score_pct": round(float(scores_row[j]) * 100, 1),
                "matched"  : ", ".join(g["matched"]),
                "missing"  : ", ".join(g["missing"]),
                "match_pct": g["match_pct"],
            })
        csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download recommendations CSV",
            data=csv_bytes,
            file_name="job_recommendations.csv",
            mime="text/csv",
        )


# EMPTY STATE

if not uploaded:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#7A9CC4">
        <p style="font-size:3rem;margin:0">📄</p>
        <p style="font-size:1.1rem;font-weight:600;margin:0.5rem 0 0.3rem;color:#0B2447">
            No resume uploaded yet
        </p>
        <p style="font-size:0.88rem;margin:0">
            Upload a PDF above to get started — skill extraction is automatic.
        </p>
    </div>""", unsafe_allow_html=True)
elif not resume_skills:
    st.warning("No skills were detected. Try a different PDF or add skills manually.")