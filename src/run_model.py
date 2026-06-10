import pandas as pd
from model import JobMatchingModel

# Load extracted resume skills (from skill_extractor output)
resumes = pd.read_csv("/home/surendran-g/Documents/AI_Job_Recommendation_System/notebooks/outputs/resume_skills.csv")

# Convert string skills → list
resume_skills = resumes["skill_str"].fillna("").apply(lambda x: x.split(", ")).tolist()

# Load jobs (you create this file)
jobs = pd.read_csv("dataset/jobs.csv")
job_skills = jobs["skills"].apply(lambda x: x.split(", ")).tolist()

job_titles = jobs["title"].tolist()

# Train model
model = JobMatchingModel(vectorizer_type="mlb")
scores = model.fit_match(resume_skills, job_skills)

# Show top match for first resume
top = model.top_k_jobs(scores, resume_idx=0, k=3, job_ids=job_titles)

print("\nTOP JOB MATCHES:")
for t in top:
    print(t)