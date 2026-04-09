import pandas as pd
import joblib
import re

# --- 1. CONFIGURATION ---
print("🚀 Loading Clinical Engines...")
try:
    model = joblib.load('dataset/final_model.pkl')
    symptom_columns = joblib.load('dataset/symptom_list.pkl')
    print("   ✅ Brain Loaded.")
except:
    print("   ⚠️  Model files not found. Running in Logic-Only mode.")
    model = None
    symptom_columns = []

# --- 2. THE MASTER PATTERN LIST (Your Provided Source) ---
# We convert your list of dicts into the format our parser needs: {Label: [patterns]}
NLP_DATA = [
    {"label": "Fever", "patterns": ["fever", "hot", "high temp", "burning up"]},
    {"label": "Cough", "patterns": ["cough", "coughing"]},
    {"label": "Chills", "patterns": ["chills", "shivering", "rigors"]},
    {"label": "Fatigue_Severe", "patterns": ["tired", "fatigue", "exhausted", "lethargic"]},
    {"label": "Shortness_of_breath", "patterns": ["short of breath", "breathless", "dyspnea", "cant breathe"]},
    {"label": "Loss_Appetite", "patterns": ["no appetite", "anorexia", "not hungry"]},
    {"label": "Sore_throat", "patterns": ["sore throat", "odynophagia", "throat hurts", "swallow"]},
    {"label": "Pain_Chest_Upper", "patterns": ["chest pain", "thoracic pain", "tight chest"]},
    {"label": "Pain_Head", "patterns": ["headache", "cephalalgia"]},
    {"label": "Pain_Muscle_Diffuse", "patterns": ["muscle pain", "myalgia", "body aches"]},
    {"label": "Runny_Nose_Clear", "patterns": ["runny nose", "rhinorrhea"]},
    {"label": "Nasal_Congestion", "patterns": ["stuffy", "congestion", "blocked nose"]},
    {"label": "Sputum_Colored", "patterns": ["sputum", "phlegm", "mucus", "green", "yellow"]},
    {"label": "Coughing_Blood", "patterns": ["blood", "hemoptysis", "coughing blood"]},
    {"label": "Wheezing", "patterns": ["wheeze", "wheezing", "stridor"]},
    {"label": "History_Asthma", "patterns": ["have asthma", "asthmatic"]},
    {"label": "History_Smoker", "patterns": ["smoke", "smoker", "tobacco"]},
    {"label": "History_COPD", "patterns": ["copd", "emphysema"]},
]

# Convert to Dictionary format for faster lookup
SYMPTOM_MAP = {item['label']: item['patterns'] for item in NLP_DATA}

def analyze_robust_transcript(text):
    # Split sentences (handles space after punctuation)
    sentences = re.split(r"(?<=[.!?]) +", text)
    
    extracted_symptoms = []
    last_question_symptoms = []  # Memory buffer
    
    # Vocabulary
    affirmations = ["yes", "yeah", "yep", "correct", "i did", "i have", "that's right", "sure", "definitely"]
    negations = ["no", "not", "never", "don't", "none", "nope"]
    question_starters = ["do", "have", "has", "is", "are", "can", "could", "any", "what"]

    print(f"📋 Processing {len(sentences)} sentences...")

    for i, sentence in enumerate(sentences):
        s = sentence.lower().strip()
        if not s: continue
        
        # --- A. IS THIS A QUESTION? (Doctor) ---
        # Ends with '?' OR starts with 'Are', 'Do', 'What', etc.
        is_question = s.endswith("?") or any(s.startswith(w + " ") for w in question_starters)
        
        if is_question:
            # 1. Start Fresh: Clear old pending questions
            last_question_symptoms = []
            
            # 2. Check for Medical Keywords
            current_q_syms = []
            for sym, patterns in SYMPTOM_MAP.items():
                if any(p in s for p in patterns):
                    current_q_syms.append(sym)
            
            if current_q_syms:
                last_question_symptoms = current_q_syms
                print(f"🤔 [Line {i}] Doctor Asked about: {last_question_symptoms}")
            continue 

        # --- B. IS THIS AN ANSWER? (Patient) ---
        if last_question_symptoms:
            # Affirmation Check
            if any(w in s for w in affirmations):
                print(f"✅ [Line {i}] Patient Confirmed: {last_question_symptoms}")
                extracted_symptoms.extend(last_question_symptoms)
                last_question_symptoms = [] # Clear logic
                continue
            
            # Denial Check
            elif any(w in s for w in negations):
                print(f"🚫 [Line {i}] Patient Denied: {last_question_symptoms}")
                last_question_symptoms = [] # Clear logic
                continue
            
            # Ambiguous (Patient ignored question and changed topic)
            else:
                last_question_symptoms = []

        # --- C. DIRECT STATEMENT (Patient) ---
        # If not answering a question, scan for direct mentions
        for sym, patterns in SYMPTOM_MAP.items():
            if any(p in s for p in patterns):
                # Ensure it's not negated (e.g. "No cough")
                if not any(n in s for n in negations):
                    print(f"📝 [Line {i}] Direct Mention: {sym}")
                    extracted_symptoms.append(sym)

    return list(set(extracted_symptoms))

# --- 3. TEST EXECUTION ---
# Use this section to paste your raw conversation text
raw_text = (
    "Hello. I'm the doctor. How can I help? "
    "I have had a bad cough. "
    "I see. Have you had a fever? "
    "I don't think so. "
    "Okay. What about your breathing? Are you short of breath? "
    "Yes, especially on the stairs." 
)

print("-" * 40)
final_list = analyze_robust_transcript(raw_text)
print("-" * 40)
print(f"\n🎯 FINAL SYMPTOMS FOR AI: {final_list}")

if model:
    input_vector = pd.DataFrame(0, index=[0], columns=symptom_columns)
    # Ensure we only use columns the model actually knows
    valid_symptoms = [s for s in final_list if s in symptom_columns]
    
    for s in valid_symptoms:
        input_vector.at[0, s] = 1
    
    if valid_symptoms:
        prediction = model.predict(input_vector)[0]
        probs = model.predict_proba(input_vector)[0]
        confidence = max(probs)
        print(f"🏥 DIAGNOSIS: {prediction} ({confidence*100:.1f}%)")
    else:
        print("🏥 DIAGNOSIS: Unknown (No valid model symptoms detected)")