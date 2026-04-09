import pandas as pd
import joblib
import numpy as np
import json

class RespiratoryDiagnosisSystem:
    def __init__(self):
        print("⚙️  Loading Inference Engine...")
        self.model = joblib.load('dataset/final_model.pkl')
        self.symptom_columns = joblib.load('dataset/symptom_list.pkl')
        self.knowledge_base = joblib.load('dataset/knowledge_base.pkl')
        
        # 1. Try loading the JSON Dictionary
        try:
            with open('dataset/release_evidences.json', 'r') as f:
                self.evidence_dict = json.load(f)
        except Exception:
            self.evidence_dict = {}

        # 2. MANUAL BACKUP DICTIONARY (The "Force Fix")
        # These are common respiratory codes that sometimes get lost
        self.manual_map = {
            'E_181': 'Sneezing',
            'E_129': 'Runny nose', 
            'E_173': 'Nasal congestion',
            'E_201': 'Cough', 
            'E_91':  'Fever',
            'E_53':  'Coughing up blood',
            'E_54':  'Sputum',
            'E_55':  'Purulent sputum (yellow/green)',
            'E_144': 'Shortness of breath',
            'E_222': 'Wheezing'
        }

    def get_nice_name(self, code):
        """Translates 'E_181' -> 'Sneezing' using all available methods"""
        # Method A: Check the JSON file
        if code in self.evidence_dict:
            return self.evidence_dict[code]['name']
        
        # Method B: Check our Manual Map (The Fix)
        if code in self.manual_map:
            return self.manual_map[code]
            
        # Method C: Fail gracefully (Return the code itself)
        return code

    def predict_disease(self, current_symptoms):
        input_vector = pd.DataFrame(0, index=[0], columns=self.symptom_columns)
        
        for sym in current_symptoms:
            if sym in self.symptom_columns:
                input_vector[sym] = 1
        
        probs = self.model.predict_proba(input_vector)[0]
        classes = self.model.classes_
        
        results = {disease: float(prob) for disease, prob in zip(classes, probs)}
        return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    def get_next_question(self, current_symptoms):
        matching_rows = self.knowledge_base.copy()
        
        for sym in current_symptoms:
            if sym in matching_rows.columns:
                matching_rows = matching_rows[matching_rows[sym] == 1]
                
        if len(matching_rows) < 2:
            return None

        best_symptom_code = None
        best_score = -1 
        
        potential_questions = [col for col in self.symptom_columns if col not in current_symptoms]
        
        for sym in potential_questions:
            count = matching_rows[sym].sum()
            total = len(matching_rows)
            if total == 0: continue
            
            probability = count / total
            score = 1 - abs(probability - 0.5) * 2
            
            if score > best_score:
                best_score = score
                best_symptom_code = sym
        
        # Translate the Code before returning!
        if best_symptom_code:
            return self.get_nice_name(best_symptom_code)
        
        return None

if __name__ == "__main__":
    system = RespiratoryDiagnosisSystem()
    
    # Test
    patient_symptoms = ['Cough', 'Fever']
    
    print(f"\n😷 Current Symptoms: {patient_symptoms}")
    
    diagnoses = system.predict_disease(patient_symptoms)
    top_disease = list(diagnoses.keys())[0]
    print(f"🤖 Top Prediction: {top_disease} ({diagnoses[top_disease]:.2f})")
    
    next_q = system.get_next_question(patient_symptoms)
    print(f"💡 Recommended Next Question: Do you have '{next_q}'?")