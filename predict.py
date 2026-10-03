from resume_utils import load_model, top_predictions
model = load_model()
text = input("Enter Resume Skills: ")
print("\nTop predictions:")
for cat, p in top_predictions(model, text):
    print(f"  {cat:20s} {p*100:5.1f}%")
