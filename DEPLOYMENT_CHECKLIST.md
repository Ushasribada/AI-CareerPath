# Permanent Deployment Checklist

### Before GitHub
- [ ] Add `training_data.csv`, `skills_list.csv`, `job_roles.csv` under `backend/data/resume_dataset/`.
- [ ] Add `job_description.csv` under `backend/data/job_market_dataset/`.
- [ ] Verify no passwords, API keys, ngrok tokens, or private resumes are included.
- [ ] Commit and push.

### Render
- Runtime: Python
- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`

### Frontend
- Change `API_BASE_URL` in `frontend/script.js` from the placeholder to the Render URL.
- Deploy `frontend/` as a static site.
- Test PDF and DOCX upload.

### Viva demo
- Keep one clean sample resume ready.
- Demonstrate: upload → extracted skills → skill gap → market trends → recommendation → explanation → roadmap.
