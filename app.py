import streamlit as st
from resume_utils import load_model, read_pdf, top_predictions, matched_skills, jd_match_score

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="centered")
model = load_model()

st.title("📄 AI Resume Screening System")
st.caption("Upload a resume or paste skills to predict the best-fit job role.")

tab1, tab2 = st.tabs(["Resume Analysis", "Job Description Match"])

with tab1:
    pdf = st.file_uploader("Upload resume (PDF)", type="pdf")
    text = st.text_area("...or paste resume / skills", height=180)
    if pdf is not None:
        text = read_pdf(pdf)
        st.info(f"PDF read successfully ({len(text.split())} words).")
    if st.button("Analyze Resume", type="primary"):
        if not text.strip():
            st.warning("Please upload a PDF or enter resume details.")
        else:
            preds = top_predictions(model, text)
            best, conf = preds[0]
            st.success(f"Best-fit role: **{best}**  ({conf*100:.1f}% confidence)")
            if conf < 0.5:
                st.warning("Low confidence: the resume may fit several roles or contain few skill keywords.")
            st.subheader("Top 3 roles")
            for cat, p in preds:
                st.write(cat); st.progress(p, text=f"{p*100:.1f}%")
            skills = matched_skills(model, text, best)
            if skills:
                st.subheader("Key skills detected")
                st.write(", ".join(f"`{s}`" for s in skills))

with tab2:
    st.write("Compare a resume with a job description.")
    resume_t = st.text_area("Resume text / skills", height=140, key="r")
    jd_t = st.text_area("Job description", height=140, key="j")
    if st.button("Calculate Match"):
        if resume_t.strip() and jd_t.strip():
            score = jd_match_score(resume_t, jd_t)
            st.metric("Match score", f"{score:.1f}%")
            st.progress(min(score / 100, 1.0))
        else:
            st.warning("Please fill both boxes.")
