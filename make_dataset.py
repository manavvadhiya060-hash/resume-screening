"""Generates a larger training dataset (resumes.csv) from skill pools.
Each resume mixes core skills of its category + a few shared/noisy skills,
so categories overlap like real resumes. Replace this with real resumes when available."""
import random, pandas as pd
random.seed(42)
CORE = {
 "Python Developer": ["Python","Django","Flask","FastAPI","SQL","REST API","Pandas","NumPy","Machine Learning","Deep Learning","TensorFlow","PyTorch","scikit-learn","Git","Docker","PostgreSQL","Unit Testing"],
 "Web Developer": ["HTML","CSS","JavaScript","TypeScript","React","Angular","Vue","NodeJS","Express","Bootstrap","Tailwind","MongoDB","REST API","Git","Responsive Design","Webpack","Next.js"],
 "Android Developer": ["Java","Kotlin","Android Studio","Firebase","XML","Jetpack Compose","Retrofit","Room Database","MVVM","Material Design","Gradle","Mobile Development","REST API","Git","SQLite"],
 "Data Analyst": ["Power BI","Excel","SQL","Tableau","Data Visualization","Dashboard","Analytics","Statistics","Python","Pandas","Data Cleaning","Reporting","Pivot Tables","ETL","Business Intelligence"],
 "UI Designer": ["UI","UX","Figma","Adobe XD","Wireframing","Prototyping","User Research","Photoshop","Illustrator","Design Systems","Typography","Color Theory","Usability Testing","Sketch","Interaction Design"],
}
SOFT = ["Communication","Teamwork","Problem Solving","Time Management","Leadership","Agile"]
rows=[]
for cat,skills in CORE.items():
    for _ in range(60):
        s = random.sample(skills, random.randint(4,8))
        if random.random()<.5: s += random.sample(SOFT, random.randint(1,2))
        if random.random()<.25:  # overlap noise from another category
            other=random.choice([c for c in CORE if c!=cat]); s.append(random.choice(CORE[other]))
        random.shuffle(s); rows.append((" ".join(s),cat))
random.shuffle(rows)
pd.DataFrame(rows,columns=["Resume","Category"]).to_csv("resumes.csv",index=False)
print("resumes.csv created:",len(rows),"rows")
