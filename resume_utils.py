import re, joblib
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip().lower()

def read_pdf(file):
    reader = PdfReader(file)
    return " ".join((p.extract_text() or "") for p in reader.pages)

def top_predictions(model, text, k=3):
    proba = model.predict_proba([clean_text(text)])[0]
    idx = np.argsort(proba)[::-1][:k]
    return [(model.classes_[i], float(proba[i])) for i in idx]

def matched_skills(model, text, category, n=8):
    """Words in the resume that the model finds most indicative of the category."""
    tfidf = model.named_steps["tfidf"]; clf = model.named_steps["clf"]
    vocab = tfidf.get_feature_names_out()
    ci = list(clf.classes_).index(category)
    if hasattr(clf, "calibrated_classifiers_"):   # calibrated SVM -> average the inner linear models
        weights = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)[ci]
    elif hasattr(clf, "coef_"):
        weights = clf.coef_[ci]
    else:
        weights = clf.feature_log_prob_[ci]
    present = tfidf.transform([clean_text(text)]).nonzero()[1]
    ranked = sorted(present, key=lambda j: weights[j], reverse=True)
    return [vocab[j] for j in ranked if " " not in vocab[j]][:n]

def jd_match_score(resume, job_description):
    v = TfidfVectorizer(ngram_range=(1, 2)).fit([clean_text(resume), clean_text(job_description)])
    m = v.transform([clean_text(resume), clean_text(job_description)])
    return float(cosine_similarity(m[0], m[1])[0][0]) * 100

def load_model(path="resume_model.pkl"):
    return joblib.load(path)
