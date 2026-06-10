"""
recommender.py
==============
AI Job Recommendation System — Job Recommender

Orchestrates the full pipeline:
  1. Load resume skills + job listings
  2. Fit JobMatchingModel on the combined vocabulary
  3. Generate cosine similarity matrix  (n_resumes × n_jobs)
  4. Expose recommendation, skill-gap, and summary APIs
  5. Persist results to CSV

Typical usage
-------------
    recommender = JobRecommender()
    recommender.load_data()
    recommender.train()

    # top-5 jobs for resume 0
    print(recommender.recommend_jobs(resume_idx=0, top_k=5))

    # skill gap between resume 0 and job 0
    print(recommender.skill_gap_analysis(resume_idx=0, job_idx=0))

    # persist all recommendations
    recommender.save_recommendations()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── project-local imports ─────────────────────────────────────────────────────
# Adjust sys.path so this module can be run from any working directory.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from model import JobMatchingModel  # noqa: E402  (after sys.path patch)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── default paths (relative to project root) ──────────────────────────────────
_PROJECT_ROOT   = _SRC_DIR.parent
_RESUME_SKILLS = _PROJECT_ROOT / "notebooks" / "outputs" / "resume_skills.csv"
_JOBS_CSV       = _PROJECT_ROOT / "dataset" / "jobs.csv"
_OUTPUT_DIR     = _PROJECT_ROOT / "outputs"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_skill_str(raw: Any) -> list[str]:
    """
    Convert a comma-separated skill string (or existing list) to a clean list.

    Examples
    --------
    >>> _parse_skill_str("Python, SQL, AWS")
    ['Python', 'SQL', 'AWS']
    >>> _parse_skill_str(float('nan'))
    []
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [s.strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class JobRecommender:
    """
    End-to-end job recommendation engine.

    Parameters
    ----------
    resume_skills_path : str | Path
        Path to ``outputs/resume_skills.csv`` produced by skill_extractor.py.
    jobs_path : str | Path
        Path to ``dataset/jobs.csv``.
    vectorizer_type : str
        Vectorisation strategy passed to JobMatchingModel.
        One of ``'mlb'``, ``'tfidf'``, ``'hybrid'``. Default ``'mlb'``.
    output_dir : str | Path
        Directory where output files are written. Default ``outputs/``.

    Attributes
    ----------
    resumes_df : pd.DataFrame
        Loaded resume-skills table after ``load_data()``.
    jobs_df : pd.DataFrame
        Loaded jobs table after ``load_data()``.
    resume_skills : list[list[str]]
        Parsed skill lists for every resume.
    job_skills : list[list[str]]
        Parsed skill lists for every job.
    model : JobMatchingModel
        Fitted matching model after ``train()``.
    scores : np.ndarray
        Cosine-similarity matrix of shape ``(n_resumes, n_jobs)``
        after ``train()``.
    """

    def __init__(
        self,
        resume_skills_path: str | Path = _RESUME_SKILLS,
        jobs_path:          str | Path = _JOBS_CSV,
        vectorizer_type:    str        = "mlb",
        output_dir:         str | Path = _OUTPUT_DIR,
    ) -> None:
        self.resume_skills_path: Path = Path(resume_skills_path)
        self.jobs_path:          Path = Path(jobs_path)
        self.vectorizer_type:    str  = vectorizer_type
        self.output_dir:         Path = Path(output_dir)

        # populated by load_data()
        self.resumes_df:    pd.DataFrame    = pd.DataFrame()
        self.jobs_df:       pd.DataFrame    = pd.DataFrame()
        self.resume_skills: list[list[str]] = []
        self.job_skills:    list[list[str]] = []

        # populated by train()
        self.model:  JobMatchingModel | None = None
        self.scores: np.ndarray | None       = None

        self._is_trained: bool = False

    # ── internal guards ───────────────────────────────────────────────────────

    def _require_data(self) -> None:
        if self.resumes_df.empty or self.jobs_df.empty:
            raise RuntimeError(
                "Data not loaded. Call load_data() before this method."
            )

    def _require_trained(self) -> None:
        if not self._is_trained or self.scores is None:
            raise RuntimeError(
                "Model not trained. Call train() before this method."
            )

    def _validate_resume_idx(self, idx: int) -> None:
        if not (0 <= idx < len(self.resumes_df)):
            raise IndexError(
                f"resume_idx={idx} is out of range "
                f"[0, {len(self.resumes_df) - 1}]."
            )

    def _validate_job_idx(self, idx: int) -> None:
        if not (0 <= idx < len(self.jobs_df)):
            raise IndexError(
                f"job_idx={idx} is out of range "
                f"[0, {len(self.jobs_df) - 1}]."
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 1.  LOAD DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def load_data(self) -> "JobRecommender":
        """
        Load resume skills and job listings from disk.

        Reads
        -----
        - ``resume_skills_path`` → expects columns: ID, Category, skill_str, n_skills
        - ``jobs_path``          → expects columns: job_id, title, category, skills

        Returns
        -------
        self  (for method chaining)

        Raises
        ------
        FileNotFoundError
            If either file does not exist.
        ValueError
            If required columns are missing.
        """
        # ── resumes ───────────────────────────────────────────────────────────
        log.info("Loading resumes from %s …", self.resume_skills_path)
        if not self.resume_skills_path.exists():
            raise FileNotFoundError(
                f"Resume skills file not found: {self.resume_skills_path}\n"
                "Run skill_extractor.py first to generate it."
            )
        self.resumes_df = pd.read_csv(self.resume_skills_path)

        _required_resume_cols = {"ID", "Category", "skill_str"}
        _missing = _required_resume_cols - set(self.resumes_df.columns)
        if _missing:
            raise ValueError(
                f"resume_skills.csv is missing columns: {_missing}. "
                "Re-run skill_extractor.py."
            )

        self.resume_skills = [
            _parse_skill_str(row)
            for row in self.resumes_df["skill_str"]
        ]
        log.info(
            "Loaded %d resumes  (avg %.1f skills/resume)",
            len(self.resumes_df),
            float(self.resumes_df["n_skills"].mean())
            if "n_skills" in self.resumes_df.columns
            else 0.0,
        )

        # ── jobs ──────────────────────────────────────────────────────────────
        log.info("Loading jobs from %s …", self.jobs_path)
        if not self.jobs_path.exists():
            raise FileNotFoundError(
                f"Jobs file not found: {self.jobs_path}\n"
                "Ensure dataset/jobs.csv exists."
            )
        self.jobs_df = pd.read_csv(self.jobs_path)

        _required_job_cols = {"job_id", "title", "skills"}
        _missing = _required_job_cols - set(self.jobs_df.columns)
        if _missing:
            raise ValueError(
                f"jobs.csv is missing columns: {_missing}."
            )

        self.job_skills = [
            _parse_skill_str(row)
            for row in self.jobs_df["skills"]
        ]
        log.info("Loaded %d job listings", len(self.jobs_df))
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # 2 + 3.  TRAIN  (fit model + generate similarity matrix)
    # ═══════════════════════════════════════════════════════════════════════════

    def train(self) -> "JobRecommender":
        """
        Fit the matching model and compute the full similarity matrix.

        Must be called after ``load_data()``.

        Steps
        -----
        1. Instantiate ``JobMatchingModel`` with the configured vectorizer.
        2. Fit on the union of resume + job skill vocabularies.
        3. Store the ``(n_resumes × n_jobs)`` cosine-similarity matrix
           in ``self.scores``.

        Returns
        -------
        self  (for method chaining)

        Raises
        ------
        RuntimeError
            If ``load_data()`` has not been called.
        """
        self._require_data()

        log.info(
            "Initialising JobMatchingModel  (vectorizer='%s') …",
            self.vectorizer_type,
        )
        self.model = JobMatchingModel(vectorizer_type=self.vectorizer_type)

        log.info("Training matching model …")
        self.scores = self.model.fit_match(self.resume_skills, self.job_skills)

        self._is_trained = True
        log.info(
            "Similarity matrix ready — shape %s  "
            "(min=%.3f  max=%.3f  mean=%.3f)",
            self.scores.shape,
            float(self.scores.min()),
            float(self.scores.max()),
            float(self.scores.mean()),
        )
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # 4.  RECOMMEND JOBS
    # ═══════════════════════════════════════════════════════════════════════════

    def recommend_jobs(
        self,
        resume_idx: int,
        top_k:      int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k best-matching jobs for a given resume.

        Parameters
        ----------
        resume_idx : int
            Zero-based index into the loaded resumes DataFrame.
        top_k : int
            Number of recommendations to return. Default 5.

        Returns
        -------
        list[dict]
            Sorted list (best first) of::

                {
                    "rank":      1,
                    "job_id":    7,
                    "job_title": "ML Engineer",
                    "category":  "INFORMATION-TECHNOLOGY",
                    "score":     0.91
                }

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        IndexError
            If ``resume_idx`` is out of range.
        ValueError
            If ``top_k`` < 1.
        """
        self._require_trained()
        self._validate_resume_idx(resume_idx)

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        k = min(top_k, len(self.jobs_df))
        job_ids    = self.jobs_df["job_id"].tolist()
        job_titles = self.jobs_df["title"].tolist()
        job_cats   = (
            self.jobs_df["category"].tolist()
            if "category" in self.jobs_df.columns
            else [""] * len(self.jobs_df)
        )

        raw = self.model.top_k_jobs(           # type: ignore[union-attr]
            scores=self.scores,
            resume_idx=resume_idx,
            k=k,
            job_ids=list(range(len(self.jobs_df))),  # use int indices internally
        )

        results: list[dict[str, Any]] = []
        for item in raw:
            j = item["job_id"]                 # integer index
            results.append(
                {
                    "rank"     : item["rank"],
                    "job_id"   : int(job_ids[j]),
                    "job_title": job_titles[j],
                    "category" : job_cats[j],
                    "score"    : round(float(item["score"]), 4),
                }
            )
        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 5.  SKILL GAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def skill_gap_analysis(
        self,
        resume_idx: int,
        job_idx:    int,
    ) -> dict[str, Any]:
        """
        Identify matched, missing, and extra skills for a resume–job pair.

        Parameters
        ----------
        resume_idx : int
            Zero-based index of the resume.
        job_idx : int
            Zero-based index of the job.

        Returns
        -------
        dict with keys:

        - **resume_id** – original ID from resume_skills.csv
        - **job_title** – job title string
        - **matched**   – skills present in both
        - **missing**   – skills the job requires but the resume lacks
        - **extra**     – skills the resume has beyond job requirements
        - **match_pct** – percentage of job skills covered (0–100)
        - **score**     – pre-computed cosine similarity

        Raises
        ------
        RuntimeError, IndexError
        """
        self._require_trained()
        self._validate_resume_idx(resume_idx)
        self._validate_job_idx(job_idx)

        gap = self.model.skill_gap(            # type: ignore[union-attr]
            resume_skills=self.resume_skills[resume_idx],
            job_skills=self.job_skills[job_idx],
        )
        return {
            "resume_id": int(self.resumes_df["ID"].iloc[resume_idx]),
            "job_title": str(self.jobs_df["title"].iloc[job_idx]),
            "matched"  : gap["matched"],
            "missing"  : gap["missing"],
            "extra"    : gap["extra"],
            "match_pct": gap["match_pct"],
            "score"    : round(float(self.scores[resume_idx, job_idx]), 4),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 6.  RESUME SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    def resume_summary(self, resume_idx: int) -> dict[str, Any]:
        """
        Return a concise profile for a single resume.

        Parameters
        ----------
        resume_idx : int

        Returns
        -------
        dict::

            {
                "resume_id": 12345,
                "category":  "INFORMATION-TECHNOLOGY",
                "skills":    ["Python", "SQL", ...],
                "n_skills":  12
            }

        Raises
        ------
        RuntimeError
            If data has not been loaded.
        IndexError
            If ``resume_idx`` is out of range.
        """
        self._require_data()
        self._validate_resume_idx(resume_idx)

        row    = self.resumes_df.iloc[resume_idx]
        skills = self.resume_skills[resume_idx]
        return {
            "resume_id": int(row["ID"]),
            "category" : str(row["Category"]),
            "skills"   : skills,
            "n_skills" : len(skills),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 7.  JOB SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    def job_summary(self, job_idx: int) -> dict[str, Any]:
        """
        Return a concise profile for a single job.

        Parameters
        ----------
        job_idx : int

        Returns
        -------
        dict::

            {
                "job_id":    7,
                "job_title": "Data Scientist",
                "category":  "INFORMATION-TECHNOLOGY",
                "skills":    ["Python", "ML", ...],
                "n_skills":  10
            }

        Raises
        ------
        RuntimeError
            If data has not been loaded.
        IndexError
            If ``job_idx`` is out of range.
        """
        self._require_data()
        self._validate_job_idx(job_idx)

        row    = self.jobs_df.iloc[job_idx]
        skills = self.job_skills[job_idx]
        return {
            "job_id"   : int(row["job_id"]),
            "job_title": str(row["title"]),
            "category" : str(row.get("category", "")),
            "skills"   : skills,
            "n_skills" : len(skills),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 8.  SAVE RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def save_recommendations(
        self,
        output_file: str | Path = "",
        top_k:       int        = 5,
    ) -> Path:
        """
        Generate top-k recommendations for every resume and persist to CSV.

        Parameters
        ----------
        output_file : str | Path
            Destination CSV path.
            Defaults to ``<output_dir>/recommendations.csv``.
        top_k : int
            Number of top jobs stored per resume. Default 5.

        Returns
        -------
        Path
            Resolved path to the saved CSV.

        CSV columns
        -----------
        resume_id, resume_category, rank, job_id,
        job_title, job_category, score

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        self._require_trained()

        out_path = (
            Path(output_file)
            if output_file
            else self.output_dir / "recommendations.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "Saving top-%d recommendations for %d resumes → %s …",
            top_k, len(self.resumes_df), out_path,
        )

        records: list[dict[str, Any]] = []
        for idx in range(len(self.resumes_df)):
            row      = self.resumes_df.iloc[idx]
            top_jobs = self.recommend_jobs(resume_idx=idx, top_k=top_k)
            for rec in top_jobs:
                records.append(
                    {
                        "resume_id"      : int(row["ID"]),
                        "resume_category": str(row["Category"]),
                        "rank"           : rec["rank"],
                        "job_id"         : rec["job_id"],
                        "job_title"      : rec["job_title"],
                        "job_category"   : rec["category"],
                        "score"          : rec["score"],
                    }
                )

        pd.DataFrame(records).to_csv(out_path, index=False)
        log.info("Recommendations saved — %d rows written.", len(records))
        return out_path

    # ═══════════════════════════════════════════════════════════════════════════
    # 9.  FULL PIPELINE CONVENIENCE
    # ═══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        top_k:       int        = 5,
        output_file: str | Path = "",
    ) -> Path:
        """
        Run the complete pipeline in one call:
        ``load_data → train → save_recommendations``.

        Parameters
        ----------
        top_k : int
            Recommendations per resume.
        output_file : str | Path
            Optional override for the output CSV path.

        Returns
        -------
        Path to the saved recommendations CSV.
        """
        return self.load_data().train().save_recommendations(
            output_file=output_file, top_k=top_k
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return (
            f"JobRecommender("
            f"resumes={len(self.resumes_df)}, "
            f"jobs={len(self.jobs_df)}, "
            f"vectorizer='{self.vectorizer_type}', "
            f"status={status})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import pprint

    SEP = "=" * 65

    # ── 1. Initialise ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  AI Job Recommendation System")
    print(SEP)

    recommender = JobRecommender(vectorizer_type="mlb")

    # ── 2. Load data ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 1 — Load data")
    print(SEP)
    recommender.load_data()
    print(recommender)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 2 — Train matching model")
    print(SEP)
    recommender.train()
    print(f"  Similarity matrix: {recommender.scores.shape}")

    # ── 4. Resume summary ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 3 — Resume summary  (idx=0)")
    print(SEP)
    r_summary = recommender.resume_summary(resume_idx=0)
    pprint.pprint(r_summary)

    # ── 5. Job summary ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 4 — Job summary  (idx=0)")
    print(SEP)
    j_summary = recommender.job_summary(job_idx=0)
    pprint.pprint(j_summary)

    # ── 6. Recommend jobs ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 5 — Top-5 job recommendations  (resume idx=0)")
    print(SEP)
    recs = recommender.recommend_jobs(resume_idx=0, top_k=5)
    for r in recs:
        print(
            f"  #{r['rank']}  score={r['score']:.4f}"
            f"  [{r['category']}]  {r['job_title']}"
        )

    # ── 7. Skill gap ──────────────────────────────────────────────────────────
    best_job_idx = int(
        np.argmax(recommender.scores[0])
    )
    print(f"\n{SEP}")
    print(
        f"  Step 6 — Skill gap  "
        f"(resume 0 vs best job: idx={best_job_idx})"
    )
    print(SEP)
    gap = recommender.skill_gap_analysis(resume_idx=0, job_idx=best_job_idx)
    print(f"  Job         : {gap['job_title']}")
    print(f"  Score       : {gap['score']:.4f}")
    print(f"  Match       : {gap['match_pct']}%")
    print(f"  Matched     : {gap['matched']}")
    print(f"  Missing     : {gap['missing']}")
    print(f"  Extra       : {gap['extra'][:5]}{'...' if len(gap['extra']) > 5 else ''}")

    # ── 8. Demo a few more resumes ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 7 — Sample recommendations (resumes 1–4)")
    print(SEP)
    for idx in range(1, min(5, len(recommender.resumes_df))):
        top1 = recommender.recommend_jobs(resume_idx=idx, top_k=1)[0]
        cat  = recommender.resumes_df["Category"].iloc[idx]
        print(
            f"  resume {idx:4d}  [{cat:<25}]  →  "
            f"{top1['job_title']:<35}  score={top1['score']:.4f}"
        )

    # ── 9. Save recommendations ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Step 8 — Save recommendations CSV")
    print(SEP)
    saved_path = recommender.save_recommendations(top_k=5)
    df_check   = pd.read_csv(saved_path)
    print(f"  File     : {saved_path}")
    print(f"  Rows     : {len(df_check)}")
    print(f"  Columns  : {df_check.columns.tolist()}")
    print(f"\n  Preview (first 5 rows):")
    print(df_check.head().to_string(index=False))

    print(f"\n{SEP}")
    print("  Pipeline complete.")
    print(SEP)