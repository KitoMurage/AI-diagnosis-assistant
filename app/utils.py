import joblib
import re
import pandas as pd
import os

# --- 1. LOAD THE AI MODELS (Safe Pathing for Flask) ---
print("🚀 Initializing Clinical Engines...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../ml_pipeline/dataset/rf_model.pkl')
SYMPTOM_PATH = os.path.join(BASE_DIR, '../ml_pipeline/dataset/symptom_list.pkl')

try:
    # Load the Random Forest (Explainable AI)
    model = joblib.load(MODEL_PATH) 
    symptom_columns = joblib.load(SYMPTOM_PATH)
    print(f"   ✅ Brain Loaded ({len(symptom_columns)} symptoms).")
except Exception as e:
    print(f"   ❌ FATAL: Could not load models. {e}")
    model = None
    symptom_columns = []

# --- 2. NLP PATTERNS (Your robust mapping) ---
NLP_PATTERNS = [
    {"label": "Fever", "patterns": ["fever", "hot", "high temp", "burning up"]},
    {"label": "Cough", "patterns": ["cough", "coughing", "hacking"]},
    {"label": "Chills", "patterns": ["chills", "shivering", "rigors", "cold sweats"]},
    {"label": "Fatigue_Severe", "patterns": ["tired", "fatigue", "exhausted", "lethargic", "get out of bed"]},
    {"label": "Shortness_of_breath", "patterns": ["short of breath", "shortness of breath", "breathless", "dyspnea", "cant breathe"]},
    {"label": "Pain_Breathing_Deeply", "patterns": ["breathe deeply", "hurt to breathe", "pain when i breathe"]},
    {"label": "Loss_Appetite", "patterns": ["no appetite", "anorexia", "not hungry"]},
    {"label": "Sore_throat", "patterns": ["sore throat", "odynophagia", "throat hurts", "swallow"]},
    {"label": "Pain_Chest_Upper", "patterns": ["chest pain", "thoracic pain", "tight chest", "chest hurts", "pain in my chest"]},
    {"label": "Pain_Head", "patterns": ["headache", "cephalalgia", "head"]},
    {"label": "Pain_Muscle_Diffuse", "patterns": ["muscle pain", "myalgia", "body aches"]},
    {"label": "Runny_Nose_Clear", "patterns": ["runny nose", "rhinorrhea", "snot"]},
    {"label": "Nasal_Congestion", "patterns": ["stuffy", "congestion", "blocked nose"]},
    {"label": "Sputum_Colored", "patterns": ["sputum", "phlegm", "mucus", "green", "yellow"]},
    {"label": "Coughing_Blood", "patterns": ["blood", "hemoptysis", "coughing blood"]},
    {"label": "Voice_Hoarseness", "patterns": ["hoarse", "lose my voice", "voice is gone"]},
    {"label": "Wheezing", "patterns": ["wheeze", "wheezing", "stridor"]},
    {"label": "History_Asthma", "patterns": ["have asthma", "asthmatic"]},
    {"label": "History_Smoker", "patterns": ["smoke", "smoker", "tobacco"]},
    {"label": "History_COPD", "patterns": ["copd", "emphysema"]},
]

# --- 3. CLINICAL LOGIC (NICE GUIDELINES) ---
DOCTOR_LOGIC = {
    'Cough': [
        ('Sputum_Colored', "Is your cough productive? (Are you bringing up any yellow or green phlegm?)"),
        ('Coughing_Blood', "Warning Sign Check: Have you coughed up any blood?"),
        ('History_Smoker', "Do you smoke?"),
        ('Shortness_of_breath', "Are you feeling breathless?")
    ],
    'Shortness_of_breath': [
        ('Wheezing', "Do you hear a wheezing or whistling sound when you breathe?"),
        ('Pain_Chest_Upper', "Do you have any pain in your chest?"),
        ('History_Asthma', "Do you have a history of Asthma?"),
        ('History_COPD', "Have you been diagnosed with COPD?")
    ],
    'Sore_throat': [
        ('Fever', "Do you have a fever?"),
        ('Runny_Nose_Clear', "Do you have a runny nose?")
    ],
    'Fever': [
        ('Chills', "Are you experiencing chills or shivering?"),
        ('Pain_Muscle_Diffuse', "Do you have body aches?")
    ]
}

# --- 4. NLP FUNCTIONS (Your exact logic) ---
def extract_symptoms_robust(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r'(Doctor:|Patient:|Dr\.:|Dr |Patient )', '', text, flags=re.IGNORECASE)
    
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    confirmed = []
    denied = []
    pending = [] 
    
    question_starters = ["do", "does", "have", "has", "is", "are", "can", "could", "any", "what"]
    affirmations = ["yes", "yeah", "yep", "correct", "i do", "i have", "right", "sure"]
    negations = ["no", "not", "never", "don't", "dont", "nope"]

    for sentence in sentences:
        s_clean = sentence.strip().lower()
        if not s_clean: continue
        
        # Step A: Resolve Pending from previous sentence
        if pending:
            is_affirmative = any(s_clean.startswith(w) or f" {w} " in f" {s_clean} " for w in affirmations)
            is_negative = any(s_clean.startswith(w) or f" {w} " in f" {s_clean} " for w in negations)
            
            if is_affirmative:
                confirmed.extend(pending)
                pending = []
            elif is_negative:
                denied.extend(pending)
                pending = []
            else:
                pending = [] # Ambiguous, drop it

        # Step B: Scan Current Sentence
        found_in_sentence = []
        for rule in NLP_PATTERNS:
            for pattern in rule["patterns"]:
                if pattern in s_clean:
                    idx = s_clean.find(pattern)
                    chunk = s_clean[max(0, idx-30):idx]
                    is_negated = any(f" {neg} " in f" {chunk} " for neg in negations)
                    if not is_negated:
                        found_in_sentence.append(rule["label"])
                    break 

        if not found_in_sentence: continue

        # Step C: Question or Statement?
        is_question = s_clean.endswith("?") or any(s_clean.startswith(q + " ") for q in question_starters)

        if is_question:
            pending.extend(found_in_sentence)
        else:
            confirmed.extend(found_in_sentence)

    return list(set(confirmed)), list(set(denied)), list(set(pending))

def get_next_question(current_symptoms, denied_symptoms):
    for symptom in current_symptoms:
        if symptom in DOCTOR_LOGIC:
            pathway = DOCTOR_LOGIC[symptom]
            for next_sym, question in pathway:
                if next_sym not in current_symptoms and next_sym not in denied_symptoms and next_sym in symptom_columns:
                    return question, next_sym

    general = [
        ('Fever', "Do you have a fever?"),
        ('Cough', "Do you have a cough?"),
        ('Shortness_of_breath', "Are you short of breath?"),
        ('History_Smoker', "Do you smoke?"),
    ]
    for sym, q in general:
        if sym not in current_symptoms and sym not in denied_symptoms and sym in symptom_columns:
            return q, sym
            
    return "I have gathered enough information. Please proceed to examination.", None

def predict_disease(symptoms_list):
    if not symptoms_list or model is None:
        return "Waiting...", 0.0

    # Ensure we use the exact vocabulary the Random Forest expects
    input_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for s in symptoms_list:
        if s in symptom_columns:
            input_vector.at[0, s] = 1
    
    probs = model.predict_proba(input_vector)[0]
    top_idx = probs.argsort()[-1]
    top_disease = model.classes_[top_idx]
    confidence = float(probs[top_idx])
    
    return top_disease, confidence