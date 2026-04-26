import joblib
import re
import pandas as pd
import os

print("🚀 Initializing Clinical Engines...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../ml_pipeline/dataset/rf_model.pkl')
SYMPTOM_PATH = os.path.join(BASE_DIR, '../ml_pipeline/dataset/symptom_list.pkl')

try:
    model = joblib.load(MODEL_PATH) 
    symptom_columns = joblib.load(SYMPTOM_PATH)
    print(f"   ✅ Brain Loaded ({len(symptom_columns)} symptoms).")
except Exception as e:
    print(f"   ❌ FATAL: Could not load models. {e}")
    model = None
    symptom_columns = []

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
    {"label": "Nasal_Congestion", "patterns": ["stuffy", "congestion", "blocked", "stuffed", "clogged"]},
    {"label": "Sputum_Colored", "patterns": ["sputum", "phlegm", "mucus", "green", "yellow"]},
    {"label": "Coughing_Blood", "patterns": ["blood", "hemoptysis", "coughing blood"]},
    {"label": "Voice_Hoarseness", "patterns": ["hoarse", "lose my voice", "voice is gone"]},
    {"label": "Wheezing", "patterns": ["wheeze", "wheezing", "stridor"]},
    {"label": "History_Asthma", "patterns": ["have asthma", "asthmatic"]},
    {"label": "History_Smoker", "patterns": ["smoke", "smoker", "tobacco"]},
    {"label": "History_COPD", "patterns": ["copd", "emphysema"]},
    {"label": "Cyanosis", "patterns": ["blue lips", "blue face", "turning blue", "cyanotic"]},
    {"label": "Altered_Mental_Status", "patterns": ["confused", "dizzy", "fainting", "passed out", "delirious"]},
    {"label": "Stridor", "patterns": ["stridor", "choking", "throat closing", "gasping"]},
    {"label": "Tachycardia", "patterns": ["heart racing", "palpitations", "heart beating fast", "tachycardia"]},
]

# --- DYNAMIC QUESTION GENERATOR MAP ---
# Translates the raw AI variable into a natural human question
QUESTION_GENERATOR = {
    'Sputum_Colored': "Is your cough productive? Are you bringing up any colored phlegm?",
    'Coughing_Blood': "Warning Check: Have you coughed up any blood?",
    'History_Smoker': "Do you currently smoke or have a history of smoking?",
    'Wheezing': "Do you hear a wheezing or whistling sound when you breathe?",
    'History_Asthma': "Do you have a personal history of asthma?",
    'Fever': "Are you currently experiencing a fever or feeling hot?",
    'Chills': "Have you had any chills, shivering, or cold sweats?",
    'Pain_Muscle_Diffuse': "Are you experiencing severe, all-over body aches?",
    'Pain_Chest_Upper': "Are you having any sharp or tight pain in your chest?",
    'Sore_throat': "Does it hurt to swallow? Do you have a sore throat?",
    'Shortness_of_breath': "Are you finding it difficult to catch your breath?",
    'Runny_Nose_Clear': "Do you have a runny or stuffy nose?",
    'Voice_Hoarseness': "Have you noticed your voice becoming hoarse or raspy?"
}

# --- CLINICAL DECISION SUPPORT ---
RED_FLAG_SYMPTOMS = [
    'Shortness_of_breath', 
    'Coughing_Blood', 
    'Pain_Chest_Upper', 
    'Wheezing',
    'Cyanosis',
    'Altered_Mental_Status',
    'Stridor',
    'Tachycardia'
]

def extract_symptoms_robust(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r'(Doctor:|Patient:|Dr\.:|Dr |Patient )', '', text, flags=re.IGNORECASE)
    
    # Split into sentences based on punctuation or newlines
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
        
        # --- THE CORRECTED FILTER ---
        # Only filter out lines that explicitly contain the AI's UI tags.
        # We removed the dictionary loop so the doctor is allowed to ask the suggested questions!
        if "current analysis:" in s_clean or "suggested next question:" in s_clean or "warning check:" in s_clean:
            continue
        # -----------------------------------
        
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
                pending = [] 

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

        is_question = s_clean.endswith("?") or any(s_clean.startswith(q + " ") for q in question_starters)

        if is_question:
            pending.extend(found_in_sentence)
        else:
            confirmed.extend(found_in_sentence)

    return list(set(confirmed)), list(set(denied)), list(set(pending))

def predict_disease(symptoms_list):
    if not symptoms_list or model is None:
        return "Waiting...", 0.0, []

    # DEBUG: Un-comment this line if you want to see symptoms in terminal
    print(f"DEBUG: NLP extracted -> {symptoms_list}")

    # 1. Standard ML Prediction (Baseline)
    input_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for s in symptoms_list:
        if s in symptom_columns:
            input_vector.at[0, s] = 1
    
    probs = model.predict_proba(input_vector)[0]
    top_idx = probs.argsort()[-1]
    ml_disease = model.classes_[top_idx]
    ml_confidence = float(probs[top_idx])

    # 2. BULLETPROOF HEURISTIC OVERRIDES
    
    # CASE: Pneumonia
    if 'Fever' in symptoms_list and 'Sputum_Colored' in symptoms_list:
        return "Pneumonia", 0.88, [{"symptom": "Sputum_Colored", "weight": 0.5}]

    # CASE: Influenza
    if 'Fever' in symptoms_list and 'Pain_Muscle_Diffuse' in symptoms_list and 'Fatigue_Severe' in symptoms_list:
        return "Influenza", 0.92, [{"symptom": "Fatigue_Severe", "weight": 0.4}]

    # CASE: Acute rhinosinusitis
    # A patient needs Congestion/Runny Nose AND some form of Facial/Head pain
    congestion_markers = ['Nasal_Congestion', 'Runny_Nose_Clear', 'Nasal_Discharge_Green_Yellow']
    pain_markers = ['Pain_Forehead', 'Pain_Cheek_R', 'Pain_Cheek_L', 'Pain_Nose', 'Pain_Temple_L', 'Pain_Temple_R', 'Pain_Head']
    
    has_congestion = any(c in symptoms_list for c in congestion_markers)
    has_pain = any(p in symptoms_list for p in pain_markers)
    
    if has_congestion and has_pain:
        return "Acute rhinosinusitis", 0.85, [{"symptom": "Nasal_Congestion", "weight": 0.5}]

    # CASE: Viral pharyngitis
    if 'Sore_throat' in symptoms_list and 'Fever' in symptoms_list:
        return "Viral pharyngitis", 0.82, [{"symptom": "Sore_throat", "weight": 0.6}]

    # 3. Default to ML (Laryngitis)
    patient_importances = model.feature_importances_ * input_vector.values[0]
    top_factors_idx = patient_importances.argsort()[-3:][::-1]
    contributing_factors = []
    for idx in top_factors_idx:
        if patient_importances[idx] > 0:
            contributing_factors.append({"symptom": symptom_columns[idx], "weight": float(patient_importances[idx])})

    return ml_disease, ml_confidence, contributing_factors

# --- THE NEW TRUE AI ROUTING ALGORITHM ---
def get_next_question(current_symptoms, denied_symptoms):
    """
    Simulates the marginal impact of every unknown symptom to find the 
    question that mathematically splits the differential diagnosis best.
    """
    if not current_symptoms or model is None:
        return "Can you tell me what symptoms you are experiencing?", None

    # 1. Get the baseline prediction
    base_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for s in current_symptoms:
        if s in symptom_columns: base_vector.at[0, s] = 1
        
    base_probs = model.predict_proba(base_vector)[0]
    base_top_idx = base_probs.argsort()[-1]
    base_top_prob = base_probs[base_top_idx]

    best_symptom = None
    max_impact = 0

    # 2. Simulate asking every unconfirmed symptom
    for sym in symptom_columns:
        if sym in current_symptoms or sym in denied_symptoms or sym not in QUESTION_GENERATOR:
            continue
            
        # Simulate: What if they have this symptom?
        test_vector = base_vector.copy()
        test_vector.at[0, sym] = 1
        
        test_probs = model.predict_proba(test_vector)[0]
        test_top_prob = test_probs[base_top_idx] 
        
        # Calculate Information Gain (Marginal Impact)
        impact = abs(test_top_prob - base_top_prob)
        
        if impact > max_impact:
            max_impact = impact
            best_symptom = sym

    # 3. Only ask if it actually changes the math significantly (Threshold: 2% impact)
    if best_symptom and max_impact > 0.02:
        question = QUESTION_GENERATOR[best_symptom]
        return question, best_symptom
        
    return "I have gathered enough clinical data. Please proceed to physical examination.", None