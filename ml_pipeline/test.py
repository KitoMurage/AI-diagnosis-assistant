import pandas as pd
import joblib
import numpy as np

print("🔍 Initializing AI Adversarial Stress Test...")

try:
    rf_model = joblib.load('ml_pipeline/dataset/rf_model.pkl')
    nn_model = joblib.load('ml_pipeline/dataset/nn_model.pkl')
    symptom_list = joblib.load('ml_pipeline/dataset/symptom_list.pkl')
    print("✅ Models and Vocabulary loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

def run_test_case(test_name, patient_symptoms):
    print(f"\n{'='*50}")
    print(f"🩺 TEST CASE: {test_name}")
    print(f"🗣️  Patient Input: {patient_symptoms}")
    print(f"{'-'*50}")
    

    input_vector = {col: 0 for col in symptom_list}
    for s in patient_symptoms:
        if s in input_vector:
            input_vector[s] = 1
        else:
            print(f"   [!] Note: '{s}' ignored (Out of AI Vocabulary)")
            
    df = pd.DataFrame([input_vector], columns=symptom_list)

    # 2. Evaluate Random Forest 
    rf_probs = rf_model.predict_proba(df)[0]
    rf_top_3_indices = np.argsort(rf_probs)[-3:][::-1]
    
    print("🧠 RANDOM FOREST (Explainable):")
    for i, idx in enumerate(rf_top_3_indices):
        disease = rf_model.classes_[idx]
        confidence = rf_probs[idx] * 100
        if i == 0:
            print(f"   ➔ PRIMARY: {disease.upper()} ({confidence:.1f}%)")
        else:
            print(f"   - Alt Diff: {disease} ({confidence:.1f}%)")

    # 3. Evaluate Neural Network (Show primary only, as it's a black box)
    nn_probs = nn_model.predict_proba(df)[0]
    nn_top_idx = np.argmax(nn_probs)
    print("\n🤖 NEURAL NETWORK (Black Box):")
    print(f"   ➔ PRIMARY: {nn_model.classes_[nn_top_idx].upper()} ({nn_probs[nn_top_idx]*100:.1f}%)")

# ==========================================
# THE ADVERSARIAL TEST BATTERY
# ==========================================

# Test 1: The "Textbook" Case (Should be near 100% confident)
run_test_case(
    "Textbook Influenza", 
    ['Fever', 'Chills', 'Fatigue_Severe', 'Pain_Muscle_Diffuse', 'Cough']
)

# Test 2: The "Silent Patient" (Barely any data. AI should show low confidence)
run_test_case(
    "Vague Symptoms (Just a Cough)", 
    ['Cough']
)

# Test 3: The "Overlapping" Case (Pneumonia and Bronchitis have similar profiles)
run_test_case(
    "Ambiguous Respiratory Infection", 
    ['Cough', 'Shortness_of_breath', 'Sputum_Colored', 'Fever']
)

# Test 4: The "Adversarial" Case (Illogical combination)
run_test_case(
    "Illogical Symptom Array", 
    ['Voice_Hoarseness', 'Pain_Muscle_Diffuse', 'Nasal_Polyps']
)

# Test 5: Out of Vocabulary (Testing if the system crashes on unknown words)
run_test_case(
    "Unknown Symptoms (Noise)", 
    ['Cough', 'Broken_Leg', 'Stomach_Ulcer']
)