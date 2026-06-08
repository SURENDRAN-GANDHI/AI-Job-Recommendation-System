"""
data_preprocessing.py
=====================
AI Job Recommendation System — Resume Dataset Preprocessing Pipeline

Covers:
  1. Load & validate
  2. Drop / rename columns
  3. HTML structural feature extraction
  4. Text cleaning  (lowercase, URL, email, punctuation, digits, whitespace)
  5. Stopword removal + lemmatisation  (spaCy)
  6. Length feature engineering
  7. Label encoding
  8. Outlier flagging
  9. Train / val / test split  (stratified)
 10. Save artefacts

Usage:
  python data_preprocessing.py

Outputs (written to ./outputs/):
  train.csv, val.csv, test.csv
  label_encoder.pkl
  preprocessing_report.json
"""

import os
import re
import json
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── optional spaCy (falls back to NLTK if not installed) ──────────────────────
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _BACKEND = "spacy"
except (ImportError, OSError):
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    for pkg in ["punkt", "wordnet", "stopwords", "omw-1.4"]:
        nltk.download(pkg, quiet=True)
    _LEMMATIZER = WordNetLemmatizer()
    _STOPWORDS  = set(stopwords.words("english"))
    _BACKEND    = "nltk"
    warnings.warn("spaCy not found — falling back to NLTK. "
                  "Install with: pip install spacy && python -m spacy download en_core_web_sm")

warnings.filterwarnings("ignore")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH    = Path("dataset/Resume.csv")          # adjust if needed
OUTPUT_DIR  = Path("outputs")
RANDOM_SEED = 42

SPLIT_RATIOS = dict(train=0.70, val=0.15, test=0.15)

# IQR multiplier for outlier flagging (1.5 = standard Tukey)
IQR_MULTIPLIER = 1.5

# Extra domain stopwords for resume text
RESUME_STOPWORDS = {
    "resume", "curriculum", "vitae", "cv", "dear", "sincerely",
    "regards", "sir", "madam", "ref", "reference", "page", "www",
    "http", "https", "com", "org", "net", "email", "phone", "mobile",
    "address", "linkedin", "github",
}

SECTION_KEYWORDS = [
    "skill", "experience", "education", "summary",
    "objective", "project", "certification", "award",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & VALIDATE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load CSV and run basic sanity checks."""
    log.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"ID", "Resume_str", "Resume_html", "Category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info("Shape: %s", df.shape)
    log.info("Categories (%d): %s", df["Category"].nunique(),
             sorted(df["Category"].unique()))

    # null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning("Null values found:\n%s", null_counts[null_counts > 0])

    # duplicate check
    dup_ids  = df.duplicated(subset="ID").sum()
    dup_text = df.duplicated(subset="Resume_str").sum()
    if dup_ids:
        log.warning("%d duplicate IDs — dropping duplicates (keep first)", dup_ids)
        df = df.drop_duplicates(subset="ID", keep="first")
    if dup_text:
        log.warning("%d duplicate Resume_str — dropping duplicates (keep first)", dup_text)
        df = df.drop_duplicates(subset="Resume_str", keep="first")

    df = df.reset_index(drop=True)
    log.info("After deduplication: %d rows", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  HTML STRUCTURAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_html_features(html: str) -> dict:
    """Pull numeric structural signals from Resume_html."""
    soup = BeautifulSoup(html, "html.parser")
    sections_found = []
    for tag in soup.find_all(["b", "strong", "h1", "h2", "h3"]):
        text = tag.get_text(strip=True).lower()
        for kw in SECTION_KEYWORDS:
            if kw in text:
                sections_found.append(kw)
                break
    return {
        "n_bold"          : len(soup.find_all(["b", "strong"])),
        "n_list_items"    : len(soup.find_all("li")),
        "n_lists"         : len(soup.find_all("ul")),
        "n_headings"      : len(soup.find_all(["h1", "h2", "h3"])),
        "n_paragraphs"    : len(soup.find_all("p")),
        "n_line_breaks"   : len(soup.find_all("br")),
        "has_table"       : int(bool(soup.find("table"))),
        "n_sections_found": len(set(sections_found)),
    }


def add_html_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Extracting HTML structural features …")
    feat_df = df["Resume_html"].apply(extract_html_features).apply(pd.Series)
    df = pd.concat([df, feat_df], axis=1)
    log.info("HTML features added: %s", feat_df.columns.tolist())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

# pre-compiled regex patterns (compile once, reuse)
_RE_URL      = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL    = re.compile(r"\S+@\S+\.\S+")
_RE_PHONE    = re.compile(r"\+?\d[\d\s\-().]{7,}\d")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_SPECIAL  = re.compile(r"[^a-z\s]")          # keep only letters + spaces
_RE_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline (order matters):
      1. Lowercase
      2. Strip residual HTML tags
      3. Remove URLs, emails, phone numbers
      4. Remove digits and punctuation
      5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_PHONE.sub(" ", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  STOPWORD REMOVAL + LEMMATISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _lemmatise_spacy(text: str) -> str:
    """Lemmatise using spaCy (preferred)."""
    doc = _NLP(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.lemma_) > 2
        and token.lemma_ not in RESUME_STOPWORDS
    ]
    return " ".join(tokens)


def _lemmatise_nltk(text: str) -> str:
    """Lemmatise using NLTK (fallback)."""
    tokens = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in _STOPWORDS
        and t not in RESUME_STOPWORDS
        and len(t) > 2
    ]
    return " ".join(tokens)


def lemmatise(text: str) -> str:
    if _BACKEND == "spacy":
        return _lemmatise_spacy(text)
    return _lemmatise_nltk(text)


def preprocess_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text → lemmatise to Resume_str, store in `text_clean`."""
    log.info("Cleaning text  (step 1/2 — regex) …")
    df["text_clean"] = df["Resume_str"].apply(clean_text)

    log.info("Lemmatising    (step 2/2 — %s, this may take ~1 min) …", _BACKEND)
    df["text_lemma"] = df["text_clean"].apply(lemmatise)

    # safety: drop rows where cleaning produces empty string
    empty_mask = df["text_lemma"].str.strip() == ""
    if empty_mask.any():
        log.warning("Dropping %d rows with empty text after cleaning", empty_mask.sum())
        df = df[~empty_mask].reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  LENGTH FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add char/word/sentence count columns on the raw Resume_str."""
    log.info("Computing length features …")
    df["char_count"]   = df["Resume_str"].str.len()
    df["word_count"]   = df["Resume_str"].str.split().str.len()
    df["sent_count"]   = df["Resume_str"].str.count(r"[.!?]+")
    df["avg_word_len"] = (df["char_count"] / df["word_count"]).round(2)

    # cleaned text length (post-lemmatisation)
    df["clean_word_count"] = df["text_lemma"].str.split().str.len()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  OUTLIER FLAGGING
# ═══════════════════════════════════════════════════════════════════════════════

def flag_outliers(df: pd.DataFrame,
                  col: str = "word_count",
                  multiplier: float = IQR_MULTIPLIER) -> pd.DataFrame:
    """Add `length_flag` column: 'ok', 'short_outlier', 'long_outlier'."""
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - multiplier * IQR
    hi  = Q3 + multiplier * IQR

    df["length_flag"] = "ok"
    df.loc[df[col] < lo, "length_flag"] = "short_outlier"
    df.loc[df[col] > hi, "length_flag"] = "long_outlier"

    counts = df["length_flag"].value_counts()
    log.info("Outlier flags  (IQR ×%.1f, bounds [%.0f, %.0f]):\n%s",
             multiplier, lo, hi, counts.to_string())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode Category → integer label. Returns df + fitted encoder."""
    log.info("Encoding labels …")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])
    log.info("Classes: %s", list(le.classes_))
    return df, le


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  DROP UNNECESSARY COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw HTML and intermediate columns to save memory."""
    cols_to_drop = [c for c in ["Resume_html", "sections", "html_text"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info("Dropped columns: %s", cols_to_drop)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataset(df: pd.DataFrame,
                  ratios: dict = SPLIT_RATIOS,
                  seed: int = RANDOM_SEED
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / val / test.
    Handles categories with very few samples (< 3) by adjusting strategy.
    """
    log.info("Splitting dataset  train=%.0f%%  val=%.0f%%  test=%.0f%% …",
             ratios["train"] * 100, ratios["val"] * 100, ratios["test"] * 100)

    # check minimum samples per class for stratification
    min_samples = df["label"].value_counts().min()
    use_stratify = min_samples >= 3
    if not use_stratify:
        log.warning("Some classes have < 3 samples — stratification disabled for test split.")

    stratify_col = df["label"] if use_stratify else None

    # first split: train vs (val + test)
    val_test_size = ratios["val"] + ratios["test"]
    df_train, df_val_test = train_test_split(
        df,
        test_size=val_test_size,
        stratify=stratify_col,
        random_state=seed,
    )

    # second split: val vs test
    relative_test = ratios["test"] / val_test_size
    stratify_vt = df_val_test["label"] if use_stratify else None
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=relative_test,
        stratify=stratify_vt,
        random_state=seed,
    )

    log.info("Split sizes — train: %d  val: %d  test: %d",
             len(df_train), len(df_val), len(df_test))
    return df_train, df_val, df_test


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  SAVE ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_artefacts(df_train: pd.DataFrame,
                   df_val:   pd.DataFrame,
                   df_test:  pd.DataFrame,
                   le:       LabelEncoder,
                   report:   dict,
                   out_dir:  Path = OUTPUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir   / "val.csv",   index=False)
    df_test.to_csv(out_dir  / "test.csv",  index=False)
    log.info("Saved train / val / test CSVs → %s", out_dir)

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    log.info("Saved label_encoder.pkl → %s", out_dir)

    with open(out_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved preprocessing_report.json → %s", out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(df: pd.DataFrame,
                 df_train: pd.DataFrame,
                 df_val:   pd.DataFrame,
                 df_test:  pd.DataFrame,
                 le:       LabelEncoder) -> dict:
    """Serialisable summary of the preprocessing run."""
    return {
        "total_rows"        : int(len(df)),
        "n_categories"      : int(df["Category"].nunique()),
        "categories"        : sorted(df["Category"].unique().tolist()),
        "class_counts"      : df["Category"].value_counts().to_dict(),
        "split_sizes"       : {
            "train": int(len(df_train)),
            "val"  : int(len(df_val)),
            "test" : int(len(df_test)),
        },
        "word_count_stats"  : {
            k: round(float(v), 1)
            for k, v in df["word_count"].describe().items()
        },
        "outlier_counts"    : df["length_flag"].value_counts().to_dict(),
        "nlp_backend"       : _BACKEND,
        "label_classes"     : list(le.classes_),
        "feature_columns"   : [
            c for c in df.columns
            if c not in ("Resume_str", "Resume_html", "text_clean")
        ],
    }


def run_pipeline(csv_path: Path = CSV_PATH) -> None:
    log.info("=" * 60)
    log.info("  Resume Preprocessing Pipeline  (backend: %s)", _BACKEND)
    log.info("=" * 60)

    # ── steps ────────────────────────────────────────────────────────
    df = load_and_validate(csv_path)        # 1. load
    df = add_html_features(df)              # 2. html features
    df = preprocess_text_column(df)         # 3+4. clean + lemmatise
    df = add_length_features(df)            # 5. length features
    df = flag_outliers(df)                  # 6. outlier flags
    df, le = encode_labels(df)              # 7. label encoding
    df = drop_columns(df)                   # 8. drop heavy columns

    df_train, df_val, df_test = split_dataset(df)  # 9. split

    report = build_report(df, df_train, df_val, df_test, le)

    save_artefacts(df_train, df_val, df_test, le, report)  # 10. save

    # ── final summary ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Pipeline complete.")
    log.info("  Columns in train.csv : %s", df_train.columns.tolist())
    log.info("  Train labels balance :\n%s",
             df_train["Category"].value_counts().to_string())
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()