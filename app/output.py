import joblib
import os

# Paths from your code
MODEL_PATH = 'ml_pipeline/dataset/rf_model.pkl'
SYMPTOM_PATH = 'ml_pipeline/dataset/symptom_list.pkl'

print("🔍 Investigating Clinical Brain...")

if os.path.exists(MODEL_PATH) and os.path.exists(SYMPTOM_PATH):
    model = joblib.load(MODEL_PATH)
    symptoms = joblib.load(SYMPTOM_PATH)
    
    print(f"\n✅ UNIQUE DISEASES ({len(model.classes_)} total):")
    for i, disease in enumerate(sorted(model.classes_)):
        print(f"  {i+1}. {disease}")
        
    print(f"\n✅ UNIQUE SYMPTOMS ({len(symptoms)} total):")
    # Symptoms are often stored as a list or Index; this handles both
    for i, sym in enumerate(sorted(list(symptoms))):
        print(f"  {i+1}. {sym}")
else:
    print("❌ Files not found. Check your paths.")