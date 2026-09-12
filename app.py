# AI CareerPath Backend
# M.Tech Project: Explainable AI-Based Skill Gap Analysis and Dynamic Career Path Recommendation Using NLP and Job Market Skill Trends

import os
import io
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from docx import Document

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RESUME_DIR = DATA_DIR / "resume_dataset"
JOB_DIR = DATA_DIR / "job_market_dataset"

app = Flask(__name__)
CORS(app)



# ===== Notebook cell 54 =====
# ============================================================
# STEP 3: LOAD PROJECT DATA INTO BACKEND
# ============================================================

import pandas as pd
import json
import re
from collections import Counter

# Load datasets
training_data = pd.read_csv(
    "RESUME_DIR.as_posix() + "/"training_data.csv"
)

skills_df = pd.read_csv(
    "RESUME_DIR.as_posix() + "/"skills_list.csv"
)

job_roles_df = pd.read_csv(
    "RESUME_DIR.as_posix() + "/"job_roles.csv"
)

job_description_df = pd.read_csv(
    "JOB_DIR.as_posix() + "/"job_description.csv"
)

# Load skill names
skill_list = (
    skills_df["Skill Name"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

print("=" * 70)
print("BACKEND DATA LOADING")
print("=" * 70)

print(f"Training resumes      : {len(training_data)}")
print(f"Available skills      : {len(skill_list)}")
print(f"Job roles             : {len(job_roles_df)}")
print(f"Job descriptions      : {len(job_description_df)}")

print("\nTraining data columns:")
print(list(training_data.columns))

print("\nJob role columns:")
print(list(job_roles_df.columns))

print("\nFirst 10 skills:")
print(skill_list[:10])

print("\n" + "=" * 70)
print("ALL DATASETS LOADED SUCCESSFULLY")
print("=" * 70)


# ===== Notebook cell 55 =====
# ============================================================
# STEP 4: NLP-BASED SKILL EXTRACTION
# ============================================================

def clean_text(text):
    """
    Clean resume/job text before skill extraction.
    """
    text = str(text).lower()

    # Keep letters, numbers and common programming symbols
    text = re.sub(r"[^a-zA-Z0-9+#./ -]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_skills(text, skill_list):
    """
    Extract known skills from resume text using
    case-insensitive exact phrase matching.
    """

    text = clean_text(text)

    found_skills = []

    for skill in skill_list:

        skill_clean = clean_text(skill)

        # Match complete words/phrases
        pattern = r"(?<!\w)" + re.escape(skill_clean) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return sorted(set(found_skills))


# ------------------------------------------------------------
# Test with one resume from the dataset
# ------------------------------------------------------------

sample_resume = training_data.iloc[0]["Resume Text"]

extracted_skills = extract_skills(
    sample_resume,
    skill_list
)

print("=" * 70)
print("NLP SKILL EXTRACTION TEST")
print("=" * 70)

print("\nSample Resume:")
print(sample_resume)

print("\nExtracted Skills:")
print(extracted_skills)

print("\nNumber of Extracted Skills:")
print(len(extracted_skills))

print("\n" + "=" * 70)
print("SKILL EXTRACTION ENGINE READY")
print("=" * 70)


# ===== Notebook cell 56 =====
# ============================================================
# STEP 5: SKILL GAP ANALYSIS ENGINE
# ============================================================

def parse_required_skills(required_skills):
    """
    Convert the pipe-separated required skills
    into a clean list.
    """

    if pd.isna(required_skills):
        return []

    skills = str(required_skills).split("|")

    return [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]


def calculate_skill_gap(candidate_skills, required_skills):
    """
    Calculate matching skills, missing skills,
    and skill match percentage.
    """

    candidate_set = {
        str(skill).strip().lower()
        for skill in candidate_skills
    }

    required_set = {
        str(skill).strip().lower()
        for skill in required_skills
    }

    matching = candidate_set.intersection(required_set)
    missing = required_set - candidate_set

    if len(required_set) > 0:
        match_percentage = (
            len(matching) / len(required_set)
        ) * 100
    else:
        match_percentage = 0

    return (
        round(match_percentage, 2),
        sorted(matching),
        sorted(missing)
    )


def analyze_skill_gap(candidate_skills, job_roles):
    """
    Compare candidate skills against every job role.
    """

    results = []

    for _, row in job_roles.iterrows():

        required_skills = parse_required_skills(
            row["Required Skills"]
        )

        match_percentage, matching, missing = (
            calculate_skill_gap(
                candidate_skills,
                required_skills
            )
        )

        results.append({
            "Job Role": row["Job Title"],
            "Category": row["Category"],
            "Skill Match (%)": match_percentage,
            "Matching Skills": matching,
            "Missing Skills": missing,
            "Required Skill Count": len(required_skills)
        })

    result_df = pd.DataFrame(results)

    # Rank roles from highest skill match to lowest
    result_df = result_df.sort_values(
        by="Skill Match (%)",
        ascending=False
    ).reset_index(drop=True)

    return result_df


# ------------------------------------------------------------
# Test with the sample candidate
# ------------------------------------------------------------

candidate_skills = extracted_skills

skill_gap_results = analyze_skill_gap(
    candidate_skills,
    job_roles_df
)

print("=" * 70)
print("SKILL GAP ANALYSIS TEST")
print("=" * 70)

print("\nCandidate Skills:")
print(candidate_skills)

print("\nTop 10 Recommended Roles Based on Skill Match:")
print(
    skill_gap_results[
        [
            "Job Role",
            "Category",
            "Skill Match (%)",
            "Matching Skills",
            "Missing Skills"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("SKILL GAP ANALYSIS ENGINE READY")
print("=" * 70)


# ===== Notebook cell 57 =====
# ============================================================
# STEP 6: JOB MARKET SKILL TREND ANALYSIS
# ============================================================

# Combine available job-description text fields
text_columns = [
    "Description",
    "Benefits",
    "Requirement",
    "Requirements"
]

for col in text_columns:
    if col in job_description_df.columns:
        job_description_df[col] = (
            job_description_df[col]
            .fillna("")
            .astype(str)
        )

job_description_df["Job Text"] = (
    job_description_df["Description"] + " " +
    job_description_df["Benefits"] + " " +
    job_description_df["Requirement"] + " " +
    job_description_df["Requirements"]
)


# ------------------------------------------------------------
# Extract skills from each job description
# ------------------------------------------------------------

job_description_df["Extracted Skills"] = (
    job_description_df["Job Text"].apply(
        lambda text: extract_skills(text, skill_list)
    )
)


# ------------------------------------------------------------
# Count skill demand
# ------------------------------------------------------------

skill_demand = Counter()

for skills in job_description_df["Extracted Skills"]:
    for skill in skills:
        skill_demand[skill] += 1


# ------------------------------------------------------------
# Create market-skill dataframe
# ------------------------------------------------------------

market_skills = pd.DataFrame(
    skill_demand.items(),
    columns=["Skill", "Job_Count"]
)

market_skills = market_skills.sort_values(
    by="Job_Count",
    ascending=False
).reset_index(drop=True)


# Calculate demand percentage
total_jobs = len(job_description_df)

market_skills["Demand_Percentage"] = (
    market_skills["Job_Count"] / total_jobs * 100
).round(2)


# ------------------------------------------------------------
# Create reusable market-demand dictionary
# ------------------------------------------------------------

market_demand_dict = dict(
    zip(
        market_skills["Skill"].str.lower(),
        market_skills["Demand_Percentage"]
    )
)


# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("=" * 70)
print("JOB MARKET SKILL TREND ANALYSIS")
print("=" * 70)

print("\nTotal Job Descriptions:", total_jobs)

print(
    "Unique Skills Found in Job Market:",
    len(market_skills)
)

print("\nTop 20 In-Demand Skills:\n")

print(
    market_skills.head(20).to_string(index=False)
)

print("\n" + "=" * 70)
print("JOB MARKET ANALYSIS ENGINE READY")
print("=" * 70)


# ===== Notebook cell 58 =====
# ============================================================
# STEP 7: DYNAMIC CAREER RECOMMENDATION ENGINE
# ============================================================

def calculate_market_score(required_skills):
    """
    Calculate average market demand of the skills
    required for a job role.
    """

    if not required_skills:
        return 0

    scores = []

    for skill in required_skills:

        skill_lower = str(skill).strip().lower()

        demand = market_demand_dict.get(
            skill_lower,
            0
        )

        scores.append(demand)

    return round(
        sum(scores) / len(scores),
        2
    )


def generate_dynamic_recommendations(
    candidate_skills,
    job_roles,
    skill_weight=0.70,
    market_weight=0.30
):
    """
    Generate career recommendations using:

    70% Skill Match
    30% Job Market Demand
    """

    skill_gap_df = analyze_skill_gap(
        candidate_skills,
        job_roles
    )

    recommendations = []

    for _, row in skill_gap_df.iterrows():

        required_skills = parse_required_skills(
            job_roles.loc[
                job_roles["Job Title"] == row["Job Role"],
                "Required Skills"
            ].iloc[0]
        )

        market_score = calculate_market_score(
            required_skills
        )

        skill_match = row["Skill Match (%)"]

        dynamic_score = (
            skill_weight * skill_match
            +
            market_weight * market_score
        )

        recommendations.append({
            "Job Role": row["Job Role"],
            "Category": row["Category"],
            "Skill Match (%)": skill_match,
            "Market Demand (%)": market_score,
            "Dynamic Score": round(dynamic_score, 2),
            "Matching Skills": row["Matching Skills"],
            "Missing Skills": row["Missing Skills"]
        })

    recommendation_df = pd.DataFrame(
        recommendations
    )

    recommendation_df = recommendation_df.sort_values(
        by="Dynamic Score",
        ascending=False
    ).reset_index(drop=True)

    return recommendation_df


# ------------------------------------------------------------
# Test Dynamic Recommendation
# ------------------------------------------------------------

dynamic_recommendations = generate_dynamic_recommendations(
    candidate_skills,
    job_roles_df
)

print("=" * 70)
print("DYNAMIC CAREER RECOMMENDATION TEST")
print("=" * 70)

print("\nCandidate Skills:")
print(candidate_skills)

print("\nTop 10 Dynamic Career Recommendations:\n")

print(
    dynamic_recommendations[
        [
            "Job Role",
            "Skill Match (%)",
            "Market Demand (%)",
            "Dynamic Score",
            "Matching Skills",
            "Missing Skills"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("DYNAMIC RECOMMENDATION ENGINE READY")
print("=" * 70)


# ===== Notebook cell 59 =====
# ============================================================
# STEP 8: EXPLAINABLE AI RECOMMENDATION LAYER
# ============================================================

def generate_explanation(row):
    """
    Generate a human-readable explanation
    for a career recommendation.
    """

    matching_skills = row["Matching Skills"]
    missing_skills = row["Missing Skills"]

    match_percentage = row["Skill Match (%)"]
    market_percentage = row["Market Demand (%)"]
    dynamic_score = row["Dynamic Score"]

    matching_count = len(matching_skills)
    missing_count = len(missing_skills)

    # Matching skill explanation
    if matching_count > 0:
        matching_text = ", ".join(
            skill.title()
            for skill in matching_skills
        )
    else:
        matching_text = "No major matching skills"

    # Missing skill explanation
    if missing_count > 0:
        missing_text = ", ".join(
            skill.title()
            for skill in missing_skills
        )
    else:
        missing_text = "No major missing skills"

    explanation = (
        f"{row['Job Role']} is recommended because "
        f"{matching_count} required skills match the candidate profile "
        f"({match_percentage:.2f}% skill match). "
        f"The average market demand of the required skills is "
        f"{market_percentage:.2f}%. "
        f"Matching skills include: {matching_text}. "
        f"To become better prepared for this role, the candidate "
        f"should develop: {missing_text}. "
        f"The resulting dynamic recommendation score is "
        f"{dynamic_score:.2f}."
    )

    return explanation


# ------------------------------------------------------------
# Generate explanations for all recommendations
# ------------------------------------------------------------

explainable_recommendations = (
    dynamic_recommendations.copy()
)

explainable_recommendations["Explanation"] = (
    explainable_recommendations.apply(
        generate_explanation,
        axis=1
    )
)


# ------------------------------------------------------------
# Display top recommendations with explanations
# ------------------------------------------------------------

print("=" * 70)
print("EXPLAINABLE AI RECOMMENDATION TEST")
print("=" * 70)

for i, row in explainable_recommendations.head(5).iterrows():

    print(f"\nRecommendation #{i + 1}")
    print("-" * 70)

    print("Job Role        :", row["Job Role"])
    print("Skill Match     :", row["Skill Match (%)"], "%")
    print("Market Demand   :", row["Market Demand (%)"], "%")
    print("Dynamic Score   :", row["Dynamic Score"])

    print("\nMatching Skills :")
    print(row["Matching Skills"])

    print("\nMissing Skills  :")
    print(row["Missing Skills"])

    print("\nExplanation:")
    print(row["Explanation"])


print("\n" + "=" * 70)
print("EXPLAINABLE AI LAYER READY")
print("=" * 70)


# ===== Notebook cell 60 =====
# ============================================================
# STEP 9: DYNAMIC SKILL / LEARNING ROADMAP
# ============================================================

def normalize_skill(skill):
    """
    Normalize skill names for matching between
    job-role requirements and market data.
    """
    return str(skill).strip().lower()


def calculate_skill_priority(
    missing_skills,
    market_demand_dict
):
    """
    Calculate learning priority for missing skills.

    Priority is based on:
    1. Skill gap occurrence
    2. Job market demand
    """

    roadmap = []

    for skill in missing_skills:

        normalized_skill = normalize_skill(skill)

        market_demand = market_demand_dict.get(
            normalized_skill,
            0
        )

        roadmap.append({
            "Missing Skill": skill,
            "Market Demand (%)": market_demand
        })

    return pd.DataFrame(roadmap)


# ------------------------------------------------------------
# Generate roadmap for top recommendation
# ------------------------------------------------------------

top_recommendation = (
    explainable_recommendations.iloc[0]
)

top_missing_skills = (
    top_recommendation["Missing Skills"]
)

roadmap_df = calculate_skill_priority(
    top_missing_skills,
    market_demand_dict
)


# ------------------------------------------------------------
# Calculate dynamic priority
# ------------------------------------------------------------

if len(roadmap_df) > 0:

    # Normalize market demand to 0-100
    max_demand = roadmap_df["Market Demand (%)"].max()

    if max_demand > 0:
        roadmap_df["Market Demand Score"] = (
            roadmap_df["Market Demand (%)"]
            / max_demand
            * 100
        )
    else:
        roadmap_df["Market Demand Score"] = 0

    # For this candidate, each missing skill occurs once.
    # Therefore the roadmap is primarily prioritized
    # by market demand.
    roadmap_df["Gap Frequency Score"] = 100

    roadmap_df["Dynamic Priority"] = (
        0.5 * roadmap_df["Gap Frequency Score"]
        +
        0.5 * roadmap_df["Market Demand Score"]
    )

    roadmap_df = roadmap_df.sort_values(
        by="Dynamic Priority",
        ascending=False
    ).reset_index(drop=True)


# ------------------------------------------------------------
# Display roadmap
# ------------------------------------------------------------

print("=" * 70)
print("DYNAMIC SKILL / LEARNING ROADMAP")
print("=" * 70)

print("\nRecommended Career Role:")
print(top_recommendation["Job Role"])

print("\nCurrent Matching Skills:")
print(top_recommendation["Matching Skills"])

print("\nSkills to Develop:")
print(top_missing_skills)

print("\nPrioritized Learning Roadmap:\n")

if len(roadmap_df) > 0:

    print(
        roadmap_df[
            [
                "Missing Skill",
                "Market Demand (%)",
                "Dynamic Priority"
            ]
        ].to_string(index=False)
    )

else:
    print("No missing skills identified.")

print("\n" + "=" * 70)
print("DYNAMIC LEARNING ROADMAP READY")
print("=" * 70)


# ===== Notebook cell 61 =====
# ============================================================
# STEP 10: CREATE MAIN /ANALYZE API ENDPOINT
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze_resume():

    try:

        # ----------------------------------------------------
        # 1. Receive request
        # ----------------------------------------------------

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received."
            }), 400

        resume_text = data.get("resume_text", "")

        if not resume_text.strip():
            return jsonify({
                "success": False,
                "error": "Resume text is empty."
            }), 400


        # ----------------------------------------------------
        # 2. NLP Skill Extraction
        # ----------------------------------------------------

        candidate_skills = extract_skills(
            resume_text,
            skill_list
        )


        # ----------------------------------------------------
        # 3. Skill Gap Analysis
        # ----------------------------------------------------

        skill_gap_results = analyze_skill_gap(
            candidate_skills,
            job_roles_df
        )


        # ----------------------------------------------------
        # 4. Dynamic Career Recommendation
        # ----------------------------------------------------

        dynamic_results = generate_dynamic_recommendations(
            candidate_skills,
            job_roles_df
        )


        # ----------------------------------------------------
        # 5. Explainable AI
        # ----------------------------------------------------

        explainable_results = (
            dynamic_results.copy()
        )

        explainable_results["Explanation"] = (
            explainable_results.apply(
                generate_explanation,
                axis=1
            )
        )


        # ----------------------------------------------------
        # 6. Top recommendation
        # ----------------------------------------------------

        top = explainable_results.iloc[0]

        recommended_role = top["Job Role"]

        matching_skills = top["Matching Skills"]

        missing_skills = top["Missing Skills"]


        # ----------------------------------------------------
        # 7. Dynamic Learning Roadmap
        # ----------------------------------------------------

        roadmap_df = calculate_skill_priority(
            missing_skills,
            market_demand_dict
        )

        if len(roadmap_df) > 0:

            max_demand = (
                roadmap_df["Market Demand (%)"].max()
            )

            if max_demand > 0:

                roadmap_df["Market Demand Score"] = (
                    roadmap_df["Market Demand (%)"]
                    / max_demand
                    * 100
                )

            else:

                roadmap_df["Market Demand Score"] = 0

            roadmap_df["Gap Frequency Score"] = 100

            roadmap_df["Dynamic Priority"] = (
                0.5 *
                roadmap_df["Gap Frequency Score"]
                +
                0.5 *
                roadmap_df["Market Demand Score"]
            )

            roadmap_df = roadmap_df.sort_values(
                by="Dynamic Priority",
                ascending=False
            )

        else:

            roadmap_df = pd.DataFrame(
                columns=[
                    "Missing Skill",
                    "Market Demand (%)",
                    "Dynamic Priority"
                ]
            )


        # ----------------------------------------------------
        # 8. Convert roadmap to JSON format
        # ----------------------------------------------------

        roadmap = []

        for _, row in roadmap_df.iterrows():

            roadmap.append({
                "skill": row["Missing Skill"],
                "market_demand": float(
                    row["Market Demand (%)"]
                ),
                "priority": float(
                    row["Dynamic Priority"]
                )
            })


        # ----------------------------------------------------
        # 9. Prepare recommendations for frontend
        # ----------------------------------------------------

        recommendations = []

        for _, row in explainable_results.head(10).iterrows():

            recommendations.append({
                "job_role": row["Job Role"],
                "category": row["Category"],
                "skill_match": float(
                    row["Skill Match (%)"]
                ),
                "market_demand": float(
                    row["Market Demand (%)"]
                ),
                "dynamic_score": float(
                    row["Dynamic Score"]
                ),
                "matching_skills": row["Matching Skills"],
                "missing_skills": row["Missing Skills"],
                "explanation": row["Explanation"]
            })


        # ----------------------------------------------------
        # 10. Final API response
        # ----------------------------------------------------

        response = {

            "success": True,

            "candidate": {
                "skills": candidate_skills,
                "skill_count": len(candidate_skills)
            },

            "recommendation": {
                "job_role": recommended_role,
                "skill_match": float(
                    top["Skill Match (%)"]
                ),
                "market_demand": float(
                    top["Market Demand (%)"]
                ),
                "dynamic_score": float(
                    top["Dynamic Score"]
                ),
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "explanation": top["Explanation"]
            },

            "learning_roadmap": roadmap,

            "top_recommendations": recommendations
        }


        return jsonify(response)


    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


print("=" * 70)
print("FLASK API ENDPOINT CREATED")
print("=" * 70)

print("\nEndpoint:")
print("POST /analyze")

print("\nPipeline:")
print("Resume → NLP → Skill Gap → Market Trends")
print("       → Recommendation → Explainability")
print("       → Learning Roadmap")

print("\nBackend integration endpoint is ready.")
print("=" * 70)


# ===== PDF/DOCX resume extraction =====
def extract_resume_text(file_bytes, filename):
    filename = filename.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    if filename.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip()).strip()
    raise ValueError("Unsupported file format. Please upload PDF or DOCX.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"success": True, "service": "AI CareerPath API"})


@app.route("/analyze-file", methods=["POST"])
def analyze_resume_file():
    try:
        if "resume" not in request.files:
            return jsonify({"success": False, "error": "No resume file uploaded."}), 400
        file = request.files["resume"]
        if not file.filename:
            return jsonify({"success": False, "error": "No file selected."}), 400
        resume_text = extract_resume_text(file.read(), file.filename)
        if not resume_text:
            return jsonify({"success": False, "error": "Could not extract text from the resume."}), 400

        candidate_skills = extract_skills(resume_text, skill_list)
        dynamic_results = generate_dynamic_recommendations(candidate_skills, job_roles_df)
        explainable_results = dynamic_results.copy()
        explainable_results["Explanation"] = explainable_results.apply(generate_explanation, axis=1)
        top = explainable_results.iloc[0]
        missing_skills = top["Missing Skills"]
        roadmap_df = calculate_skill_priority(missing_skills, market_demand_dict)
        roadmap=[]
        if len(roadmap_df):
            max_demand=roadmap_df["Market Demand (%)"].max()
            roadmap_df["Market Demand Score"]=(roadmap_df["Market Demand (%)"] / max_demand * 100) if max_demand>0 else 0
            roadmap_df["Gap Frequency Score"]=100
            roadmap_df["Dynamic Priority"]=0.5*roadmap_df["Gap Frequency Score"]+0.5*roadmap_df["Market Demand Score"]
            roadmap_df=roadmap_df.sort_values("Dynamic Priority",ascending=False)
            for _,row in roadmap_df.iterrows():
                roadmap.append({"skill":row["Missing Skill"],"market_demand":float(row["Market Demand (%)"]),"priority":float(row["Dynamic Priority"])})
        recommendations=[]
        for _,row in explainable_results.head(10).iterrows():
            recommendations.append({"job_role":row["Job Role"],"category":row["Category"],"skill_match":float(row["Skill Match (%)"]),"market_demand":float(row["Market Demand (%)"]),"dynamic_score":float(row["Dynamic Score"]),"matching_skills":row["Matching Skills"],"missing_skills":row["Missing Skills"],"explanation":row["Explanation"]})
        return jsonify({"success":True,"candidate":{"skills":candidate_skills,"skill_count":len(candidate_skills)},"recommendation":{"job_role":top["Job Role"],"skill_match":float(top["Skill Match (%)"]),"market_demand":float(top["Market Demand (%)"]),"dynamic_score":float(top["Dynamic Score"]),"matching_skills":top["Matching Skills"],"missing_skills":missing_skills,"explanation":top["Explanation"]},"learning_roadmap":roadmap,"top_recommendations":recommendations})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500


@app.route("/market-trends", methods=["GET"])
def get_market_trends():
    try:
        trends=market_skills.head(15)
        return jsonify({"success":True,"total_jobs":int(total_jobs),"trends":[{"skill":str(r["Skill"]),"job_count":int(r["Job_Count"]),"demand_percentage":float(r["Demand_Percentage"])} for _,r in trends.iterrows()]})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}),500


if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
