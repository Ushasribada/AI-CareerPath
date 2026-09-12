# AI CareerPath — M.Tech Project Submission Package

**Title:** Explainable AI-Based Skill Gap Analysis and Dynamic Career Path Recommendation Using NLP and Job Market Skill Trends

## Project components
- `frontend/` — HTML, CSS and JavaScript dashboard
- `backend/` — Flask REST API prepared for Render deployment
- `backend/data/` — place the project datasets here
- `research/AI_CareerPath_sanitized_notebook.ipynb` — notebook with secrets removed

## Current validated results
- Resume skill extraction: Precision 96.13%, Recall 85.51%, F1 89.66% on 10,000 resumes.
- Final recommendation test set: 2,000 resumes.
- Baseline skill-match recommendation: Hit@1 22.65%, Hit@3 35.85%, Hit@5 41.90%, MRR 0.3224.
- Corrected technical-skill model: Hit@1 19.05%, Hit@3 29.20%, Hit@5 34.05%, MRR 0.2584.
- Average skill match: 28.49%; average skill gap: 71.51%.
- Strong recommendation coverage in corrected model: 22.10%.

## Important research interpretation
The notebook shows that adding the job-market score with the tested 70/30 weighting did **not** improve recommendation accuracy over the skill-match baseline. Therefore the paper must present job-market demand as a dynamic decision-support signal and report the baseline comparison honestly; it must not claim that the 70/30 model improves Hit@1.

## Deployment order
1. Put the required CSV datasets into `backend/data/`.
2. Push this folder to GitHub.
3. Deploy `backend/` to Render.
4. Copy the Render URL into `frontend/script.js` as `API_BASE_URL`.
5. Deploy `frontend/` to Netlify or GitHub Pages.
6. Test `/health`, `/market-trends`, and resume upload end-to-end.

## Security
The original Colab notebook contained an ngrok authentication token. The submission notebook included here has that token redacted. **Rotate/revoke the exposed token before using ngrok again. Never commit an API token to GitHub.**
