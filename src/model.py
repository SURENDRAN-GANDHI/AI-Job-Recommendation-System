import logging
import pickle
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



# CONSTANTS

VECTORIZER_MLB   = "mlb"       # MultiLabelBinarizer — fast, interpretable
VECTORIZER_TFIDF = "tfidf"     # TF-IDF on joined skill strings — soft weighting
VECTORIZER_HYBRID = "hybrid"   # MLB + TF-IDF concatenated — best coverage

SUPPORTED_VECTORIZERS = {VECTORIZER_MLB, VECTORIZER_TFIDF, VECTORIZER_HYBRID}



# HELPER — skill normalisation

def _normalise_skill(skill: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return " ".join(skill.lower().strip().split())


def _normalise_skill_list(skills: list[str]) -> list[str]:
    """Deduplicate and normalise a list of skill strings."""
    seen, out = set(), []
    for s in skills:
        if not isinstance(s, str):
            continue
        norm = _normalise_skill(s)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _skills_to_str(skills: list[str]) -> str:
    """Join skill list into a single string for TF-IDF input."""
    return " ".join(s.replace(" ", "_") for s in skills)



# CORE CLASS

class JobMatchingModel:
    

    def __init__(
        self,
        vectorizer_type:    str  = VECTORIZER_MLB,
        tfidf_max_features: int  = 3000,
        normalize_vectors:  bool = True,
    ) -> None:
        if vectorizer_type not in SUPPORTED_VECTORIZERS:
            raise ValueError(
                f"vectorizer_type must be one of {SUPPORTED_VECTORIZERS}, "
                f"got '{vectorizer_type}'"
            )

        self.vectorizer_type    = vectorizer_type
        self.tfidf_max_features = tfidf_max_features
        self.normalize_vectors  = normalize_vectors

        # fitted artefacts (set during fit)
        self._mlb:   Optional[MultiLabelBinarizer] = None
        self._tfidf: Optional[TfidfVectorizer]     = None

        self.is_fitted_   = False
        self.vocabulary_: list[str] = []
        self.n_features_: int       = 0

        log.info(
            "JobMatchingModel initialised  (vectorizer=%s  normalise=%s)",
            vectorizer_type, normalize_vectors,
        )

    # private helpers

    def _validate_skill_lists(
        self, skills_list: list[list[str]], label: str = "input"
    ) -> list[list[str]]:
        """
        Validate and normalise a list-of-skill-lists.
        Returns cleaned version; raises TypeError on wrong input.
        """
        if not isinstance(skills_list, (list, tuple)):
            raise TypeError(
                f"{label} must be a list of lists, got {type(skills_list).__name__}"
            )
        cleaned = []
        for i, skills in enumerate(skills_list):
            if skills is None or (isinstance(skills, float) and np.isnan(skills)):
                cleaned.append([])
            elif isinstance(skills, str):
                # accept comma-separated string as fallback
                cleaned.append(_normalise_skill_list(
                    [s.strip() for s in skills.split(",") if s.strip()]
                ))
            elif isinstance(skills, (list, tuple, set)):
                cleaned.append(_normalise_skill_list(list(skills)))
            else:
                log.warning("Row %d: unexpected skill type %s — skipping", i, type(skills))
                cleaned.append([])
        return cleaned

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Model is not fitted. Call fit() before transform() or match()."
            )

    def _fit_mlb(self, all_skills: list[list[str]]) -> None:
        self._mlb = MultiLabelBinarizer(sparse_output=False)
        self._mlb.fit(all_skills)
        self.vocabulary_ = list(self._mlb.classes_)

    def _fit_tfidf(self, all_skills: list[list[str]]) -> None:
        corpus = [_skills_to_str(s) for s in all_skills]
        self._tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            token_pattern=r"[^\s]+",   # treat each underscore-joined skill as a token
        )
        self._tfidf.fit(corpus)
        self.vocabulary_ = list(self._tfidf.get_feature_names_out())

    def _transform_mlb(self, skills_list: list[list[str]]) -> np.ndarray:
        """
        Transform with MLB — unseen skills are silently ignored
        (MLB's default behaviour when classes are fixed at fit time).
        """
        return self._mlb.transform(skills_list).astype(np.float32)

    def _transform_tfidf(self, skills_list: list[list[str]]) -> np.ndarray:
        corpus = [_skills_to_str(s) for s in skills_list]
        mat = self._tfidf.transform(corpus)
        if issparse(mat):
            mat = mat.toarray()
        return mat.astype(np.float32)

    def _maybe_normalise(self, matrix: np.ndarray) -> np.ndarray:
        if self.normalize_vectors:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0          # avoid divide-by-zero for empty rows
            return matrix / norms
        return matrix

    # public API 

    def fit(
        self,
        resume_skills: list[list[str]],
        job_skills:    list[list[str]],
    ) -> "JobMatchingModel":
        """
        Fit the vectoriser on the union of resume + job skill vocabularies.

        Parameters
        ----------
        resume_skills : list[list[str]]
            One list of skill strings per resume.
        job_skills : list[list[str]]
            One list of skill strings per job description.

        Returns
        -------
        self
        """
        log.info(
            "Fitting model on %d resumes + %d jobs …",
            len(resume_skills), len(job_skills),
        )

        resume_clean = self._validate_skill_lists(resume_skills, "resume_skills")
        job_clean    = self._validate_skill_lists(job_skills,    "job_skills")
        all_skills   = resume_clean + job_clean

        if self.vectorizer_type == VECTORIZER_MLB:
            self._fit_mlb(all_skills)
            self.n_features_ = len(self.vocabulary_)

        elif self.vectorizer_type == VECTORIZER_TFIDF:
            self._fit_tfidf(all_skills)
            self.n_features_ = len(self.vocabulary_)

        elif self.vectorizer_type == VECTORIZER_HYBRID:
            self._fit_mlb(all_skills)
            self._fit_tfidf(all_skills)
            self.n_features_ = len(self._mlb.classes_) + len(
                self._tfidf.get_feature_names_out()
            )
            self.vocabulary_ = (
                list(self._mlb.classes_)
                + list(self._tfidf.get_feature_names_out())
            )

        self.is_fitted_ = True
        log.info(
            "Fit complete — vocabulary: %d skills  feature_dim: %d",
            len(set(self.vocabulary_)), self.n_features_,
        )
        return self

    def transform(self, skills_list: list[list[str]]) -> np.ndarray:
        """
        Convert a list of skill lists into a numeric feature matrix.

        Unseen skills (not present in the fitted vocabulary) are
        silently ignored — no KeyError is raised.

        Parameters
        ----------
        skills_list : list[list[str]]
            Each element is a list of skill strings for one resume/job.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            L2-normalised if normalize_vectors=True.
        """
        self._check_fitted()
        cleaned = self._validate_skill_lists(skills_list)

        if self.vectorizer_type == VECTORIZER_MLB:
            matrix = self._transform_mlb(cleaned)

        elif self.vectorizer_type == VECTORIZER_TFIDF:
            matrix = self._transform_tfidf(cleaned)

        elif self.vectorizer_type == VECTORIZER_HYBRID:
            mlb_mat   = self._transform_mlb(cleaned)
            tfidf_mat = self._transform_tfidf(cleaned)
            matrix    = np.hstack([mlb_mat, tfidf_mat])

        matrix = self._maybe_normalise(matrix)
        log.debug("Transformed %d samples → shape %s", len(cleaned), matrix.shape)
        return matrix

    def match(
        self,
        resume_matrix: np.ndarray,
        job_matrix:    np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between every resume and every job.

        Parameters
        ----------
        resume_matrix : np.ndarray, shape (n_resumes, n_features)
            Output of transform() on resume skill lists.
        job_matrix : np.ndarray, shape (n_jobs, n_features)
            Output of transform() on job skill lists.

        Returns
        -------
        np.ndarray of shape (n_resumes, n_jobs)
            scores[i, j] = cosine similarity between resume i and job j.
            Values in [0.0, 1.0].

        Raises
        ------
        ValueError
            If the feature dimensions of resume_matrix and job_matrix differ.
        """
        self._check_fitted()

        if resume_matrix.ndim == 1:
            resume_matrix = resume_matrix.reshape(1, -1)
        if job_matrix.ndim == 1:
            job_matrix = job_matrix.reshape(1, -1)

        if resume_matrix.shape[1] != job_matrix.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: "
                f"resume_matrix has {resume_matrix.shape[1]} features, "
                f"job_matrix has {job_matrix.shape[1]} features. "
                f"Both must be transformed by the same fitted model."
            )

        scores = cosine_similarity(resume_matrix, job_matrix)
        log.debug(
            "Similarity matrix shape: %s  (min=%.3f  max=%.3f  mean=%.3f)",
            scores.shape, scores.min(), scores.max(), scores.mean(),
        )
        return scores.astype(np.float32)

    # convenience end-to-end method 

    def fit_match(
        self,
        resume_skills: list[list[str]],
        job_skills:    list[list[str]],
    ) -> np.ndarray:
        """
        Convenience: fit on both corpora, then return the full similarity matrix.

        Equivalent to:
            model.fit(resume_skills, job_skills)
            r_mat = model.transform(resume_skills)
            j_mat = model.transform(job_skills)
            return model.match(r_mat, j_mat)

        Returns
        -------
        np.ndarray of shape (n_resumes, n_jobs)
        """
        self.fit(resume_skills, job_skills)
        r_mat = self.transform(resume_skills)
        j_mat = self.transform(job_skills)
        return self.match(r_mat, j_mat)

    # top-k retrieval helpers 

    def top_k_jobs(
        self,
        scores:   np.ndarray,
        resume_idx: int,
        k:        int = 5,
        job_ids:  Optional[list] = None,
    ) -> list[dict]:
        """
        Return the top-k jobs for a given resume index.

        Parameters
        ----------
        scores : np.ndarray, shape (n_resumes, n_jobs)
        resume_idx : int
        k : int
        job_ids : list, optional
            Human-readable job identifiers. Defaults to integer indices.

        Returns
        -------
        list of dicts: [{rank, job_id, score}, …]
        """
        row = scores[resume_idx]
        top_idx = np.argsort(row)[::-1][:k]
        results = []
        for rank, j in enumerate(top_idx, start=1):
            results.append({
                "rank"  : rank,
                "job_id": job_ids[j] if job_ids else int(j),
                "score" : round(float(row[j]), 4),
            })
        return results

    def top_k_resumes(
        self,
        scores:     np.ndarray,
        job_idx:    int,
        k:          int = 5,
        resume_ids: Optional[list] = None,
    ) -> list[dict]:
        """
        Return the top-k resumes for a given job index.

        Parameters
        ----------
        scores : np.ndarray, shape (n_resumes, n_jobs)
        job_idx : int
        k : int
        resume_ids : list, optional
            Human-readable resume identifiers. Defaults to integer indices.

        Returns
        -------
        list of dicts: [{rank, resume_id, score}, …]
        """
        col = scores[:, job_idx]
        top_idx = np.argsort(col)[::-1][:k]
        results = []
        for rank, r in enumerate(top_idx, start=1):
            results.append({
                "rank"     : rank,
                "resume_id": resume_ids[r] if resume_ids else int(r),
                "score"    : round(float(col[r]), 4),
            })
        return results

    # skill gap analysis 

    def skill_gap(
        self,
        resume_skills: list[str],
        job_skills:    list[str],
    ) -> dict:
        """
        Identify matched, missing, and extra skills between one resume and one job.

        Parameters
        ----------
        resume_skills : list[str]
            Skills from a single resume.
        job_skills : list[str]
            Skills required by a single job description.

        Returns
        -------
        dict with keys:
            matched  — skills present in both
            missing  — skills in job but not in resume (gaps to address)
            extra    — skills in resume but not required by job
            match_pct — percentage of job skills covered by the resume
        """
        resume_set = set(_normalise_skill_list(resume_skills))
        job_set    = set(_normalise_skill_list(job_skills))

        matched = sorted(resume_set & job_set)
        missing = sorted(job_set - resume_set)
        extra   = sorted(resume_set - job_set)
        pct     = round(len(matched) / len(job_set) * 100, 1) if job_set else 0.0

        return {
            "matched"   : matched,
            "missing"   : missing,
            "extra"     : extra,
            "match_pct" : pct,
        }

    # model persistence

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the fitted model to a .pkl file.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g., 'outputs/job_matching_model.pkl').
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JobMatchingModel":
        """
        Load a previously saved model from a .pkl file.

        Parameters
        ----------
        path : str or Path
            Path to the saved .pkl file.

        Returns
        -------
        JobMatchingModel (fitted)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        log.info("Model loaded ← %s  (features=%d)", path, model.n_features_)
        return model

    # diagnostics 

    def summary(self) -> dict:
        """
        Return a JSON-serialisable summary of the model's current state.
        """
        return {
            "is_fitted"        : self.is_fitted_,
            "vectorizer_type"  : self.vectorizer_type,
            "normalize_vectors": self.normalize_vectors,
            "n_features"       : self.n_features_,
            "vocabulary_size"  : len(self.vocabulary_),
            "sample_vocabulary": self.vocabulary_[:20] if self.vocabulary_ else [],
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "unfitted"
        return (
            f"JobMatchingModel("
            f"vectorizer='{self.vectorizer_type}', "
            f"n_features={self.n_features_}, "
            f"status={status})"
        )



# QUICK SMOKE TEST


if __name__ == "__main__":
    import pprint

    # sample data 
    sample_resumes = [
        ["Python", "Machine Learning", "TensorFlow", "SQL", "Docker", "Git"],
        ["Java", "Spring Boot", "Microservices", "Kubernetes", "PostgreSQL"],
        ["React", "JavaScript", "CSS", "Node.js", "GraphQL", "Figma"],
        ["Data Analysis", "Python", "Pandas", "Tableau", "SQL", "Statistics"],
        ["AWS", "Terraform", "Docker", "Kubernetes", "CI/CD", "Linux"],
    ]

    sample_jobs = [
        ["Python", "Machine Learning", "TensorFlow", "AWS", "SQL"],
        ["Java", "Spring Boot", "REST API", "MySQL", "Docker"],
        ["React", "TypeScript", "Node.js", "CSS", "REST API"],
        ["Data Analysis", "Python", "Power BI", "SQL", "Statistics"],
        ["AWS", "DevOps", "Docker", "Kubernetes", "CI/CD"],
        ["Python", "NLP", "spaCy", "Machine Learning", "PostgreSQL"],
    ]

    job_titles = [
        "ML Engineer",
        "Backend Java Dev",
        "Frontend Engineer",
        "Data Analyst",
        "DevOps Engineer",
        "NLP Engineer",
    ]

    resume_ids = [f"Resume_{i+1}" for i in range(len(sample_resumes))]

    # test all three vectorizer types 
    for vtype in [VECTORIZER_MLB, VECTORIZER_TFIDF, VECTORIZER_HYBRID]:
        print(f"\n{'='*60}")
        print(f"  Vectorizer: {vtype.upper()}")
        print(f"{'='*60}")

        model = JobMatchingModel(vectorizer_type=vtype)
        scores = model.fit_match(sample_resumes, sample_jobs)

        print(f"  Similarity matrix shape: {scores.shape}")
        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  Model: {model}")
        print()

        # top-2 jobs for Resume_1
        top_jobs = model.top_k_jobs(scores, resume_idx=0, k=3, job_ids=job_titles)
        print(f"  Top 3 jobs for {resume_ids[0]}:")
        for r in top_jobs:
            print(f"    #{r['rank']:d}  {r['job_id']:<25}  score={r['score']:.4f}")

        # skill gap for Resume_1 vs ML Engineer job
        gap = model.skill_gap(sample_resumes[0], sample_jobs[0])
        print(f"\n  Skill gap — {resume_ids[0]} vs '{job_titles[0]}':")
        print(f"    Match      : {gap['match_pct']}%")
        print(f"    Matched    : {gap['matched']}")
        print(f"    Missing    : {gap['missing']}")
        print(f"    Extra      : {gap['extra']}")

    # save / load round-trip 
    print(f"\n{'='*60}")
    print("  Save / Load round-trip")
    print(f"{'='*60}")

    model = JobMatchingModel(vectorizer_type=VECTORIZER_MLB)
    model.fit(sample_resumes, sample_jobs)
    model.save("outputs/job_matching_model.pkl")

    loaded_model = JobMatchingModel.load("outputs/job_matching_model.pkl")
    print(f"  Loaded: {loaded_model}")

    r_mat = loaded_model.transform(sample_resumes)
    j_mat = loaded_model.transform(sample_jobs)
    scores_loaded = loaded_model.match(r_mat, j_mat)
    print(f"  Scores from loaded model match original: "
          f"{np.allclose(model.fit_match(sample_resumes, sample_jobs), scores_loaded)}")

    # model summary 
    print(f"\n{'='*60}")
    print("  Model Summary")
    print(f"{'='*60}")
    pprint.pprint(loaded_model.summary())