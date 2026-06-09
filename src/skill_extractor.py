"""
skill_extractor.py
==================
AI Job Recommendation System — Resume Skill Extractor

Extracts technical and soft skills from resume text using:
  1. Rule-based keyword matching     (curated skill taxonomy)
  2. Pattern-based extraction        (regex for versions, certifications)
  3. spaCy NER + noun-chunk fallback (unseen / domain terms)
  4. TF-IDF top-N per category       (data-driven discovery)

Outputs (written to ./outputs/):
  skill_matrix.csv          — one row per resume, one col per skill (binary)
  skill_frequencies.csv     — skill name, total count, per-category counts
  skill_report.json         — summary statistics
  resume_skills.csv         — resume ID + Category + extracted skill list

Usage:
  # standalone
  python skill_extractor.py

  # imported
  from skill_extractor import SkillExtractor
  se = SkillExtractor()
  skills = se.extract("Proficient in Python, SQL and AWS. Led Agile sprints.")
"""

import re
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# ── optional spaCy ────────────────────────────────────────────────────────────
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _SPACY_OK = True
except (ImportError, OSError):
    _SPACY_OK = False
    warnings.warn(
        "spaCy not available — NER extraction disabled. "
        "Install: pip install spacy && python -m spacy download en_core_web_sm"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  SKILL TAXONOMY
#     Organised by domain → category → [skill aliases]
#     Aliases are matched case-insensitively; the FIRST alias is the
#     canonical name stored in the output.
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_TAXONOMY: dict[str, dict[str, list[str]]] = {

    # ── Programming Languages ─────────────────────────────────────────────────
    "programming_languages": {
        "Python"      : ["python"],
        "Java"        : ["java"],
        "JavaScript"  : ["javascript", "js"],
        "TypeScript"  : ["typescript", "ts"],
        "C++"         : ["c++", "cpp", "c plus plus"],
        "C#"          : ["c#", "csharp", "c sharp"],
        "C"           : [r"\bc\b"],
        "R"           : [r"\br\b", "r language", "r programming"],
        "Go"          : [r"\bgo\b", "golang"],
        "Rust"        : ["rust"],
        "Kotlin"      : ["kotlin"],
        "Swift"       : ["swift"],
        "Scala"       : ["scala"],
        "PHP"         : ["php"],
        "Ruby"        : ["ruby"],
        "Perl"        : ["perl"],
        "MATLAB"      : ["matlab"],
        "Shell/Bash"  : ["bash", "shell script", "shell scripting", r"\bsh\b"],
        "PowerShell"  : ["powershell"],
        "VBA"         : ["vba"],
        "COBOL"       : ["cobol"],
        "Fortran"     : ["fortran"],
        "Assembly"    : ["assembly language", r"\basm\b"],
    },

    # ── Web & Frontend ────────────────────────────────────────────────────────
    "web_frontend": {
        "HTML"        : ["html", "html5"],
        "CSS"         : ["css", "css3"],
        "React"       : ["react", "reactjs", "react.js"],
        "Angular"     : ["angular", "angularjs", "angular.js"],
        "Vue.js"      : ["vue", "vuejs", "vue.js"],
        "Next.js"     : ["next.js", "nextjs"],
        "jQuery"      : ["jquery"],
        "Bootstrap"   : ["bootstrap"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "SASS/SCSS"   : ["sass", "scss"],
        "Redux"       : ["redux"],
        "GraphQL"     : ["graphql"],
        "REST API"    : ["rest api", "restful", "rest services", "restful api"],
        "WebSockets"  : ["websocket", "websockets"],
        "Webpack"     : ["webpack"],
    },

    # ── Backend & Frameworks ──────────────────────────────────────────────────
    "backend_frameworks": {
        "Django"      : ["django"],
        "Flask"       : ["flask"],
        "FastAPI"     : ["fastapi"],
        "Spring Boot" : ["spring boot", "springboot"],
        "Spring"      : ["spring framework", r"\bspring\b"],
        "Node.js"     : ["node.js", "nodejs", "node js"],
        "Express.js"  : ["express.js", "expressjs", "express js"],
        "Laravel"     : ["laravel"],
        "ASP.NET"     : ["asp.net", "aspnet", "asp net"],
        ".NET"        : [r"\.net\b", "dotnet"],
        "Ruby on Rails": ["ruby on rails", "rails"],
        "Hibernate"   : ["hibernate"],
        "Microservices": ["microservices", "micro services", "microservice architecture"],
        "GraphQL"     : ["graphql"],
        "gRPC"        : ["grpc"],
    },

    # ── Databases ─────────────────────────────────────────────────────────────
    "databases": {
        "SQL"         : [r"\bsql\b"],
        "MySQL"       : ["mysql"],
        "PostgreSQL"  : ["postgresql", "postgres"],
        "SQLite"      : ["sqlite"],
        "Oracle DB"   : ["oracle database", "oracle db", r"\boracle\b"],
        "SQL Server"  : ["sql server", "mssql", "microsoft sql server"],
        "MongoDB"     : ["mongodb", "mongo"],
        "Redis"       : ["redis"],
        "Cassandra"   : ["cassandra", "apache cassandra"],
        "Elasticsearch": ["elasticsearch", "elastic search"],
        "DynamoDB"    : ["dynamodb"],
        "Firebase"    : ["firebase"],
        "Neo4j"       : ["neo4j"],
        "Hive"        : ["apache hive", r"\bhive\b"],
        "HBase"       : ["hbase"],
        "CouchDB"     : ["couchdb"],
        "InfluxDB"    : ["influxdb"],
        "NoSQL"       : ["nosql", "no-sql"],
    },

    # ── Cloud & DevOps ────────────────────────────────────────────────────────
    "cloud_devops": {
        "AWS"         : ["aws", "amazon web services"],
        "Azure"       : ["azure", "microsoft azure"],
        "GCP"         : ["gcp", "google cloud", "google cloud platform"],
        "Docker"      : ["docker"],
        "Kubernetes"  : ["kubernetes", "k8s"],
        "Terraform"   : ["terraform"],
        "Ansible"     : ["ansible"],
        "Jenkins"     : ["jenkins"],
        "CI/CD"       : ["ci/cd", "cicd", "continuous integration", "continuous deployment",
                         "continuous delivery"],
        "GitHub Actions": ["github actions"],
        "GitLab CI"   : ["gitlab ci", "gitlab-ci"],
        "Helm"        : ["helm"],
        "Prometheus"  : ["prometheus"],
        "Grafana"     : ["grafana"],
        "Nginx"       : ["nginx"],
        "Apache"      : ["apache http", "apache server"],
        "Linux"       : ["linux", "ubuntu", "centos", "debian", "rhel"],
        "Unix"        : ["unix"],
        "Vagrant"     : ["vagrant"],
        "OpenShift"   : ["openshift"],
        "Serverless"  : ["serverless", "lambda functions", "aws lambda"],
    },

    # ── Data Science & ML ─────────────────────────────────────────────────────
    "data_science_ml": {
        "Machine Learning"   : ["machine learning", r"\bml\b"],
        "Deep Learning"      : ["deep learning", r"\bdl\b"],
        "NLP"                : ["nlp", "natural language processing",
                                "text mining", "text analytics"],
        "Computer Vision"    : ["computer vision", "image recognition",
                                "object detection"],
        "TensorFlow"         : ["tensorflow"],
        "PyTorch"            : ["pytorch"],
        "Keras"              : ["keras"],
        "Scikit-learn"       : ["scikit-learn", "sklearn", "scikit learn"],
        "Pandas"             : ["pandas"],
        "NumPy"              : ["numpy"],
        "Matplotlib"         : ["matplotlib"],
        "Seaborn"            : ["seaborn"],
        "Plotly"             : ["plotly"],
        "XGBoost"            : ["xgboost"],
        "LightGBM"           : ["lightgbm"],
        "Random Forest"      : ["random forest"],
        "SVM"                : ["svm", "support vector machine", "support vector machines"],
        "Neural Networks"    : ["neural network", "neural networks", "ann", "cnn", "rnn",
                                "lstm", "transformer"],
        "Hugging Face"       : ["hugging face", "huggingface", "transformers library"],
        "OpenCV"             : ["opencv", "open cv"],
        "NLTK"               : ["nltk"],
        "spaCy"              : ["spacy"],
        "Statistics"         : ["statistics", "statistical analysis", "statistical modelling"],
        "Data Analysis"      : ["data analysis", "data analytics", "data analyst"],
        "Data Visualisation" : ["data visualization", "data visualisation", "data viz"],
        "Feature Engineering": ["feature engineering", "feature extraction",
                                "feature selection"],
        "A/B Testing"        : ["a/b testing", "ab testing", "split testing"],
        "Forecasting"        : ["forecasting", "time series", "time-series"],
        "Reinforcement Learning": ["reinforcement learning", r"\brl\b"],
    },

    # ── Big Data ──────────────────────────────────────────────────────────────
    "big_data": {
        "Hadoop"      : ["hadoop", "apache hadoop"],
        "Spark"       : ["apache spark", r"\bspark\b"],
        "Kafka"       : ["kafka", "apache kafka"],
        "Flink"       : ["apache flink", r"\bflink\b"],
        "Airflow"     : ["airflow", "apache airflow"],
        "dbt"         : [r"\bdbt\b", "data build tool"],
        "Snowflake"   : ["snowflake"],
        "Databricks"  : ["databricks"],
        "Redshift"    : ["redshift", "amazon redshift"],
        "BigQuery"    : ["bigquery", "google bigquery"],
        "ETL"         : [r"\betl\b", "extract transform load"],
        "Data Warehouse": ["data warehouse", "data warehousing", "dwh"],
        "Data Lake"   : ["data lake"],
        "Data Pipeline": ["data pipeline", "data pipelines"],
        "HDFS"        : ["hdfs"],
    },

    # ── Version Control & Collaboration ───────────────────────────────────────
    "version_control": {
        "Git"         : [r"\bgit\b"],
        "GitHub"      : ["github"],
        "GitLab"      : ["gitlab"],
        "Bitbucket"   : ["bitbucket"],
        "SVN"         : ["svn", "subversion"],
        "Jira"        : ["jira"],
        "Confluence"  : ["confluence"],
        "Trello"      : ["trello"],
        "Slack"       : ["slack"],
    },

    # ── Testing ───────────────────────────────────────────────────────────────
    "testing": {
        "Unit Testing"    : ["unit testing", "unit test", "unit tests"],
        "Integration Testing": ["integration testing", "integration test"],
        "Selenium"        : ["selenium"],
        "PyTest"          : ["pytest"],
        "JUnit"           : ["junit"],
        "Jest"            : [r"\bjest\b"],
        "Cypress"         : ["cypress"],
        "Postman"         : ["postman"],
        "Test Automation" : ["test automation", "automated testing", "automation testing"],
        "TDD"             : [r"\btdd\b", "test driven development",
                             "test-driven development"],
        "BDD"             : [r"\bbdd\b", "behavior driven development",
                             "behaviour driven development"],
        "Load Testing"    : ["load testing", "jmeter", "j-meter"],
        "Manual Testing"  : ["manual testing"],
        "QA"              : [r"\bqa\b", "quality assurance"],
    },

    # ── Networking & Security ─────────────────────────────────────────────────
    "networking_security": {
        "Network Security"  : ["network security", "cybersecurity", "cyber security",
                               "information security"],
        "Firewall"          : ["firewall"],
        "VPN"               : [r"\bvpn\b"],
        "TCP/IP"            : ["tcp/ip", "tcp ip"],
        "DNS"               : [r"\bdns\b"],
        "HTTPS/TLS"         : ["https", "tls", "ssl"],
        "Penetration Testing": ["penetration testing", "pen testing", "pentesting"],
        "OWASP"             : ["owasp"],
        "SIEM"              : [r"\bsiem\b"],
        "Wireshark"         : ["wireshark"],
        "Ethical Hacking"   : ["ethical hacking", "ethical hacker"],
        "PKI"               : [r"\bpki\b", "public key infrastructure"],
        "IAM"               : [r"\biam\b", "identity and access management"],
        "SOC"               : [r"\bsoc\b", "security operations center"],
    },

    # ── Project Management & Methodologies ───────────────────────────────────
    "project_management": {
        "Agile"       : ["agile", "agile methodology"],
        "Scrum"       : ["scrum"],
        "Kanban"      : ["kanban"],
        "Waterfall"   : ["waterfall"],
        "PMP"         : [r"\bpmp\b", "project management professional"],
        "Prince2"     : ["prince2"],
        "SAFe"        : [r"\bsafe\b", "scaled agile"],
        "Six Sigma"   : ["six sigma"],
        "ITIL"        : [r"\bitil\b"],
        "PMO"         : [r"\bpmo\b", "project management office"],
        "Risk Management": ["risk management"],
        "Stakeholder Management": ["stakeholder management", "stakeholder engagement"],
        "Budget Management": ["budget management", "budgeting"],
        "Resource Planning": ["resource planning", "resource management"],
    },

    # ── Business & Finance ────────────────────────────────────────────────────
    "business_finance": {
        "Financial Analysis"  : ["financial analysis", "financial modelling",
                                 "financial modeling"],
        "Accounting"          : ["accounting", "bookkeeping"],
        "Auditing"            : ["auditing", "internal audit"],
        "Taxation"            : ["taxation", "tax planning", "gst", "income tax"],
        "Tally"               : ["tally", "tally erp"],
        "SAP"                 : [r"\bsap\b"],
        "ERP"                 : [r"\berp\b", "enterprise resource planning"],
        "QuickBooks"          : ["quickbooks"],
        "Financial Reporting" : ["financial reporting", "financial statements"],
        "Budget Forecasting"  : ["budget forecasting", "budget planning"],
        "Equity Research"     : ["equity research"],
        "Investment Banking"  : ["investment banking"],
        "Valuation"           : ["valuation", "business valuation"],
        "Risk Analysis"       : ["risk analysis"],
        "Compliance"          : ["compliance", "regulatory compliance"],
        "KYC/AML"             : ["kyc", "aml", "know your customer", "anti money laundering"],
    },

    # ── Healthcare & Medical ──────────────────────────────────────────────────
    "healthcare": {
        "Patient Care"        : ["patient care", "patient management"],
        "Clinical Research"   : ["clinical research", "clinical trials"],
        "Medical Coding"      : ["medical coding", "icd", "cpt codes"],
        "Pharmacology"        : ["pharmacology"],
        "Anatomy"             : ["anatomy"],
        "Physiology"          : ["physiology"],
        "Nursing"             : ["nursing", "nurse"],
        "EMR/EHR"             : ["emr", "ehr", "electronic health record",
                                 "electronic medical record"],
        "Radiology"           : ["radiology", "x-ray", "mri", "ct scan"],
        "Surgery"             : ["surgery", "surgical"],
        "First Aid"           : ["first aid", "cpr", "bls"],
        "Healthcare Management": ["healthcare management", "hospital administration"],
    },

    # ── Design & Creative ─────────────────────────────────────────────────────
    "design_creative": {
        "Photoshop"       : ["photoshop", "adobe photoshop"],
        "Illustrator"     : ["illustrator", "adobe illustrator"],
        "InDesign"        : ["indesign", "adobe indesign"],
        "Figma"           : ["figma"],
        "Sketch"          : [r"\bsketch\b"],
        "XD"              : ["adobe xd"],
        "After Effects"   : ["after effects", "adobe after effects"],
        "Premiere Pro"    : ["premiere pro", "adobe premiere"],
        "UI/UX Design"    : ["ui/ux", "ui ux", "user interface", "user experience",
                             "ux design", "ui design"],
        "Wireframing"     : ["wireframing", "wireframe", "prototyping"],
        "3D Modelling"    : ["3d modeling", "3d modelling", "blender", "autocad",
                             "solidworks", "catia"],
        "Video Editing"   : ["video editing", "video production"],
        "Photography"     : ["photography"],
        "Typography"      : ["typography"],
        "Brand Design"    : ["branding", "brand design", "brand identity"],
    },

    # ── Sales & Marketing ─────────────────────────────────────────────────────
    "sales_marketing": {
        "Digital Marketing" : ["digital marketing"],
        "SEO"               : [r"\bseo\b", "search engine optimization",
                               "search engine optimisation"],
        "SEM"               : [r"\bsem\b", "search engine marketing",
                               "google ads", "ppc"],
        "Social Media"      : ["social media marketing", "social media management",
                               "social media"],
        "Content Marketing" : ["content marketing", "content strategy"],
        "Email Marketing"   : ["email marketing"],
        "CRM"               : [r"\bcrm\b", "customer relationship management",
                               "salesforce", "hubspot"],
        "Lead Generation"   : ["lead generation", "lead gen"],
        "Sales Strategy"    : ["sales strategy", "business development",
                               "b2b sales", "b2c sales"],
        "Market Research"   : ["market research", "market analysis"],
        "Brand Management"  : ["brand management"],
        "Product Marketing" : ["product marketing", "go-to-market", "gtm"],
        "Analytics"         : ["google analytics", "web analytics",
                               "marketing analytics"],
        "Copywriting"       : ["copywriting", "copy writing"],
    },

    # ── Soft Skills ───────────────────────────────────────────────────────────
    "soft_skills": {
        "Leadership"          : ["leadership", "team lead", "led a team",
                                 "led the team"],
        "Communication"       : ["communication", "verbal communication",
                                 "written communication"],
        "Teamwork"            : ["teamwork", "team player", "collaborative"],
        "Problem Solving"     : ["problem solving", "problem-solving",
                                 "analytical thinking"],
        "Critical Thinking"   : ["critical thinking"],
        "Time Management"     : ["time management"],
        "Adaptability"        : ["adaptability", "adaptable"],
        "Creativity"          : ["creativity", "creative thinking"],
        "Attention to Detail" : ["attention to detail"],
        "Multitasking"        : ["multitasking", "multi-tasking"],
        "Negotiation"         : ["negotiation", "negotiating"],
        "Presentation"        : ["presentation skills", "public speaking",
                                 "presentations"],
        "Mentoring"           : ["mentoring", "mentorship", "coaching"],
        "Cross-functional"    : ["cross-functional", "cross functional"],
        "Client Management"   : ["client management", "client relations",
                                 "client handling"],
    },

    # ── Certifications ────────────────────────────────────────────────────────
    "certifications": {
        "AWS Certified"       : ["aws certified", "aws certification"],
        "Google Certified"    : ["google certified", "google professional"],
        "Azure Certified"     : ["azure certified", "microsoft certified"],
        "CPA"                 : [r"\bcpa\b", "certified public accountant"],
        "CFA"                 : [r"\bcfa\b", "chartered financial analyst"],
        "CISSP"               : [r"\bcissp\b"],
        "CEH"                 : [r"\bceh\b", "certified ethical hacker"],
        "PMP Certified"       : ["pmp certified"],
        "Scrum Master"        : ["scrum master", "csm", "certified scrum master"],
        "Six Sigma Certified" : ["six sigma certified", "six sigma black belt",
                                 "six sigma green belt"],
        "CCNA"                : [r"\bccna\b"],
        "CCNP"                : [r"\bccnp\b"],
        "CompTIA"             : ["comptia", "security+", "network+"],
        "TOGAF"               : [r"\btogaf\b"],
    },
}

# ── flatten taxonomy to {alias_pattern → canonical_name, domain} ──────────────
def _build_lookup(taxonomy: dict) -> list[tuple[re.Pattern, str, str]]:
    """
    Returns list of (compiled_pattern, canonical_skill, domain).
    Patterns are case-insensitive and wrapped with word boundaries where safe.
    """
    lookup = []
    for domain, skills in taxonomy.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                # if alias already contains regex metacharacters → use as-is
                is_regex = any(c in alias for c in r"\.+*?[](){}^$|")
                if is_regex:
                    pattern = re.compile(alias, re.IGNORECASE)
                else:
                    safe = re.escape(alias)
                    pattern = re.compile(rf"\b{safe}\b", re.IGNORECASE)
                lookup.append((pattern, canonical, domain))
    return lookup


_SKILL_LOOKUP = _build_lookup(SKILL_TAXONOMY)


# ── certification pattern (catches "AWS Certified Solutions Architect", etc.) ─
_CERT_PATTERN = re.compile(
    r"\b(certified|certification|certificate)\s+[\w\s]{3,40}\b"
    r"|\b[\w\s]{3,30}\s+(certified|certification|certificate)\b",
    re.IGNORECASE,
)

# ── version pattern (Python 3.x, Java 11, Node 18, etc.) ─────────────────────
_VERSION_PATTERN = re.compile(
    r"\b(python|java|node|php|ruby|go|scala|kotlin|swift|angular|react|vue)\s*[\d]+[\d.]*\b",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CORE EXTRACTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SkillExtractor:
    """
    Multi-strategy skill extractor.

    Parameters
    ----------
    use_ner : bool
        Whether to use spaCy NER for unseen terms (default True if spaCy available).
    min_skill_length : int
        Minimum character length for NER-discovered skills (default 3).
    """

    def __init__(self, use_ner: bool = True, min_skill_length: int = 3):
        self.use_ner         = use_ner and _SPACY_OK
        self.min_skill_len   = min_skill_length
        self._lookup         = _SKILL_LOOKUP
        log.info("SkillExtractor ready  (NER=%s  backend=%s)",
                 self.use_ner, "spaCy" if _SPACY_OK else "regex-only")

    # ── strategy 1: keyword matching ─────────────────────────────────────────
    def _extract_keywords(self, text: str) -> list[dict]:
        found = []
        seen_canonical = set()
        for pattern, canonical, domain in self._lookup:
            if pattern.search(text) and canonical not in seen_canonical:
                found.append({"skill": canonical, "domain": domain,
                              "method": "keyword"})
                seen_canonical.add(canonical)
        return found

    # ── strategy 2: pattern-based (versions, certifications) ─────────────────
    def _extract_patterns(self, text: str) -> list[dict]:
        found = []
        for m in _VERSION_PATTERN.finditer(text):
            found.append({"skill": m.group(0).strip(),
                          "domain": "programming_languages",
                          "method": "version_pattern"})
        for m in _CERT_PATTERN.finditer(text):
            cert = m.group(0).strip()
            if len(cert) > 8:                      # filter noise
                found.append({"skill": cert,
                              "domain": "certifications",
                              "method": "cert_pattern"})
        return found

    # ── strategy 3: spaCy NER noun-chunk discovery ────────────────────────────
    def _extract_ner(self, text: str) -> list[dict]:
        if not self.use_ner:
            return []
        doc   = _NLP(text[:50_000])               # spaCy limit guard
        found = []
        # named entities tagged as ORG or PRODUCT often capture tech names
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "GPE"} \
               and len(ent.text) >= self.min_skill_len:
                found.append({"skill": ent.text.strip(),
                              "domain": "ner_discovered",
                              "method": "ner"})
        # noun chunks: short (2–4 tokens), mostly noun/adj — catch unseen terms
        for chunk in doc.noun_chunks:
            tokens = [t for t in chunk if not t.is_stop and not t.is_punct]
            if 1 <= len(tokens) <= 4:
                phrase = chunk.text.strip()
                if len(phrase) >= self.min_skill_len:
                    found.append({"skill": phrase,
                                  "domain": "ner_discovered",
                                  "method": "noun_chunk"})
        return found

    # ── public API ────────────────────────────────────────────────────────────
    def extract(self, text: str, include_ner: bool = False) -> list[dict]:
        """
        Extract skills from a single text string.

        Returns list of dicts: [{skill, domain, method}, ...]
        Keyword matches are always included; NER/noun-chunk results are
        included only when include_ner=True (adds latency).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        results  = self._extract_keywords(text)
        results += self._extract_patterns(text)
        if include_ner:
            results += self._extract_ner(text)

        # deduplicate by canonical skill name (keep first occurrence)
        seen, unique = set(), []
        for r in results:
            key = r["skill"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def extract_skill_names(self, text: str, include_ner: bool = False) -> list[str]:
        """Convenience wrapper — returns only the skill name strings."""
        return [r["skill"] for r in self.extract(text, include_ner=include_ner)]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skills_dataframe(df: pd.DataFrame,
                             text_col: str = "Resume_str",
                             id_col:   str = "ID",
                             cat_col:  str = "Category",
                             extractor: SkillExtractor | None = None
                             ) -> pd.DataFrame:
    """
    Apply skill extraction to every row in df.

    Returns a new DataFrame with columns:
      ID, Category, skills (list), n_skills, skill_str (comma-joined)
    """
    if extractor is None:
        extractor = SkillExtractor()

    log.info("Extracting skills from %d resumes …", len(df))
    records = []
    for i, row in df.iterrows():
        skills = extractor.extract_skill_names(str(row[text_col]))
        records.append({
            id_col    : row[id_col],
            cat_col   : row[cat_col],
            "skills"  : skills,
            "n_skills": len(skills),
            "skill_str": ", ".join(skills),
        })
        if (i + 1) % 500 == 0:
            log.info("  … processed %d / %d", i + 1, len(df))

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SKILL FREQUENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_skill_frequencies(skill_df: pd.DataFrame,
                              cat_col: str = "Category") -> pd.DataFrame:
    """
    Build a frequency table: skill × category counts + total.

    Returns DataFrame with columns:
      skill | total | <cat1> | <cat2> | … | domain
    """
    log.info("Computing skill frequencies …")

    # explode list column → one row per (resume, skill)
    exploded = skill_df.explode("skills").dropna(subset=["skills"])
    exploded = exploded[exploded["skills"].str.strip() != ""]

    # total counts
    total = exploded["skills"].value_counts().rename("total")

    # per-category counts
    cat_counts = (
        exploded.groupby(["skills", cat_col])
        .size()
        .unstack(fill_value=0)
    )

    freq_df = cat_counts.join(total).reset_index()
    freq_df = freq_df.rename(columns={"skills": "skill"})
    freq_df = freq_df.sort_values("total", ascending=False).reset_index(drop=True)

    # attach domain info
    skill_to_domain = {}
    for domain, skills in SKILL_TAXONOMY.items():
        for canonical in skills:
            skill_to_domain[canonical] = domain
    freq_df["domain"] = freq_df["skill"].map(skill_to_domain).fillna("other")

    return freq_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SKILL MATRIX  (binary, resume × skill)
# ═══════════════════════════════════════════════════════════════════════════════

def build_skill_matrix(skill_df: pd.DataFrame,
                       id_col:  str = "ID",
                       min_skill_freq: int = 5) -> pd.DataFrame:
    """
    Build a binary resume × skill matrix.

    Parameters
    ----------
    min_skill_freq : int
        Drop skills appearing in fewer than this many resumes (reduces sparsity).

    Returns
    -------
    DataFrame with ID as index, one column per skill (0/1).
    """
    log.info("Building skill matrix (min_freq=%d) …", min_skill_freq)

    exploded = skill_df[[id_col, "skills"]].explode("skills")
    exploded = exploded[exploded["skills"].str.strip() != ""]

    # filter rare skills
    skill_counts = exploded["skills"].value_counts()
    common_skills = skill_counts[skill_counts >= min_skill_freq].index.tolist()
    exploded = exploded[exploded["skills"].isin(common_skills)]

    # pivot to binary matrix
    matrix = (
        exploded.assign(present=1)
        .pivot_table(index=id_col, columns="skills",
                     values="present", aggfunc="max", fill_value=0)
    )
    matrix.columns.name = None
    log.info("Skill matrix shape: %s  (%d skills)", matrix.shape, matrix.shape[1])
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TF-IDF SKILL DISCOVERY  (data-driven top-N per category)
# ═══════════════════════════════════════════════════════════════════════════════

def tfidf_top_skills_per_category(df: pd.DataFrame,
                                   text_col: str = "Resume_str",
                                   cat_col:  str = "Category",
                                   top_n:    int = 15) -> dict[str, list[str]]:
    """
    Use TF-IDF to discover the most discriminating terms per category.
    These complement the keyword-matched skills with data-driven signals.

    Returns dict: {category: [top_n_terms]}
    """
    log.info("Running TF-IDF skill discovery (top %d per category) …", top_n)

    results = {}
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df[text_col].astype(str))
    terms = vectorizer.get_feature_names_out()

    for category in sorted(df[cat_col].unique()):
        mask  = (df[cat_col] == category).values
        if mask.sum() == 0:
            continue
        mean_tfidf = X[mask].mean(axis=0).A1
        top_idx    = mean_tfidf.argsort()[::-1][:top_n]
        results[category] = [terms[i] for i in top_idx]

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def build_skill_report(skill_df: pd.DataFrame,
                       freq_df:  pd.DataFrame,
                       tfidf_skills: dict) -> dict:
    total_skills_extracted = skill_df["n_skills"].sum()
    return {
        "total_resumes"       : int(len(skill_df)),
        "total_skills_extracted": int(total_skills_extracted),
        "avg_skills_per_resume" : round(float(skill_df["n_skills"].mean()), 2),
        "max_skills_per_resume" : int(skill_df["n_skills"].max()),
        "min_skills_per_resume" : int(skill_df["n_skills"].min()),
        "unique_skills_found"   : int(len(freq_df)),
        "top_20_skills_overall" : freq_df.head(20)["skill"].tolist(),
        "tfidf_top_per_category": {k: v[:5] for k, v in tfidf_skills.items()},
        "skills_by_domain"      : (
            freq_df.groupby("domain")["total"].sum()
            .sort_values(ascending=False).to_dict()
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(skill_df:    pd.DataFrame,
                 freq_df:     pd.DataFrame,
                 matrix:      pd.DataFrame,
                 report:      dict,
                 out_dir:     Path = Path("outputs")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume-level skills
    skill_df.drop(columns=["skills"], errors="ignore").to_csv(
        out_dir / "resume_skills.csv", index=False
    )
    log.info("Saved resume_skills.csv")

    # frequency table
    freq_df.to_csv(out_dir / "skill_frequencies.csv", index=False)
    log.info("Saved skill_frequencies.csv")

    # binary matrix (can be large — warn if > 10 MB)
    matrix_path = out_dir / "skill_matrix.csv"
    matrix.to_csv(matrix_path)
    size_mb = matrix_path.stat().st_size / 1_048_576
    log.info("Saved skill_matrix.csv  (%.1f MB)", size_mb)

    # json report
    with open(out_dir / "skill_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved skill_report.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(csv_path: Path = Path("dataset/Resume.csv"),
                 out_dir:  Path = Path("outputs"),
                 min_skill_freq: int = 5) -> None:

    log.info("=" * 60)
    log.info("  Skill Extraction Pipeline")
    log.info("=" * 60)

    # load — use preprocessed file if available, else raw CSV
    preprocessed = out_dir / "train.csv"
    if preprocessed.exists():
        log.info("Loading preprocessed train.csv …")
        df = pd.read_csv(preprocessed)
        text_col = "text_clean" if "text_clean" in df.columns else "Resume_str"
    else:
        log.info("Preprocessed file not found — loading raw CSV …")
        df = pd.read_csv(csv_path)
        text_col = "Resume_str"

    extractor = SkillExtractor()

    # step 1 — extract per resume
    skill_df = extract_skills_dataframe(df, text_col=text_col, extractor=extractor)

    # step 2 — frequency table
    freq_df = compute_skill_frequencies(skill_df)

    # step 3 — binary matrix
    matrix = build_skill_matrix(skill_df, min_skill_freq=min_skill_freq)

    # step 4 — TF-IDF discovery
    tfidf_skills = tfidf_top_skills_per_category(df, text_col=text_col)

    # step 5 — report
    report = build_skill_report(skill_df, freq_df, tfidf_skills)

    # step 6 — save
    save_outputs(skill_df, freq_df, matrix, report, out_dir)

    # ── print summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Pipeline complete.")
    log.info("  Avg skills / resume : %.1f", report["avg_skills_per_resume"])
    log.info("  Unique skills found : %d",   report["unique_skills_found"])
    log.info("  Top 10 skills overall:")
    for skill in report["top_20_skills_overall"][:10]:
        row = freq_df[freq_df["skill"] == skill]
        if not row.empty:
            log.info("    %-30s  %d resumes", skill, row["total"].values[0])
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()