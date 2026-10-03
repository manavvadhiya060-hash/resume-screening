import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from resume_utils import clean_text

df = pd.read_csv("resumes.csv")
df["Resume"] = df["Resume"].apply(clean_text)
X, y = df["Resume"], df["Category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}\n")

def pipe(clf): return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)), ("clf", clf)])
candidates = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=3),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = {}
for name, clf in candidates.items():
    s = cross_val_score(pipe(clf), X_train, y_train, cv=cv)
    scores[name] = s.mean()
    print(f"{name:20s} 5-fold CV accuracy: {s.mean()*100:.2f}% (+/- {s.std()*100:.2f})")

best = max(scores, key=scores.get)
print(f"\nBest model: {best}")
model = pipe(candidates[best]).fit(X_train, y_train)
pred = model.predict(X_test)
print("\nTest-set report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred, labels=model.classes_))
joblib.dump(model, "resume_model.pkl")
print("\nModel saved: resume_model.pkl")
