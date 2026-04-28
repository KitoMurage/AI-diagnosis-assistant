import unittest
import json
import pandas as pd
from unittest.mock import patch
import numpy as np
from app import create_app 
from extensions import db 
from models import Doctor, Patient, Consultation
from utils import extract_symptoms_robust, predict_disease, get_next_question, RED_FLAG_SYMPTOMS


# PART 1: CLINICAL ENGINE UNIT TESTS
class TestClinicalEngine(unittest.TestCase):

    # --- NLP EXTRACTION ---
    def test_extract_basic_symptoms(self):
        text = "I have a terrible headache and a fever."
        confirmed, denied, pending = extract_symptoms_robust(text)
        self.assertIn("Pain_Head", confirmed)
        self.assertIn("Fever", confirmed)
        self.assertEqual(len(denied), 0)
        self.assertEqual(len(pending), 0)

    def test_extract_negations(self):
        text = "I have a cough, but I do not have a fever."
        confirmed, denied, pending = extract_symptoms_robust(text)
        self.assertIn("Cough", confirmed)
        self.assertNotIn("Fever", confirmed) 

    def test_extract_pending_question(self):
        text = "Do you have any shortness of breath?"
        confirmed, denied, pending = extract_symptoms_robust(text)
        self.assertIn("Shortness_of_breath", pending)
        self.assertNotIn("Shortness_of_breath", confirmed)

    def test_extract_ui_tag_filtering(self):
        text = "Current Analysis: Pneumonia. Suggested Next Question: Do you have a fever?"
        confirmed, denied, pending = extract_symptoms_robust(text)
        self.assertNotIn("Fever", pending)
        self.assertNotIn("Fever", confirmed)

    # --- HEURISTIC OVERRIDES ---
    @patch('utils.model')
    def test_heuristic_pneumonia(self, mock_model):
        disease, confidence, factors = predict_disease(['Fever', 'Sputum_Colored'])
        self.assertEqual(disease, "Pneumonia")
        self.assertEqual(confidence, 0.88)
        self.assertEqual(factors[0]["symptom"], "Sputum_Colored")

    @patch('utils.model')
    def test_heuristic_rhinosinusitis(self, mock_model):
        disease, confidence, factors = predict_disease(['Nasal_Congestion', 'Pain_Forehead'])
        self.assertEqual(disease, "Acute rhinosinusitis")
        self.assertEqual(confidence, 0.85)

    # --- INFORMATION GAIN ---
    @patch('utils.model')
    def test_get_next_question_threshold(self, mock_model):
        # Mock predict_proba to simulate NO significant mathematical change (< 2% impact)
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.5, 0.1, 0.1]])
        question, sym_tag = get_next_question(['Cough'], [])
        self.assertIn("gathered enough clinical data", question)
        self.assertIsNone(sym_tag)

    def test_red_flag_constants(self):
        self.assertIn('Cyanosis', RED_FLAG_SYMPTOMS)
        self.assertIn('Altered_Mental_Status', RED_FLAG_SYMPTOMS)



# PART 2: SYSTEM INTEGRATION TESTS (Tests Flask routes, DB, and Security)
class TestSystemIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a blank, in-memory database and test client using the app factory."""
        # --- THE FIXED SETUP ---
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        
        # Override the database URI to use a temporary in-memory database
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        db.create_all()

        # Seed Clinicians
        self.doc1 = Doctor(username="dr_smith", department="Pulmonology")
        self.doc1.set_password("password123")
        self.doc2 = Doctor(username="dr_jones", department="General Practice")
        self.doc2.set_password("password123")
        db.session.add_all([self.doc1, self.doc2])
        db.session.commit()

        # Seed Patient & Consultation
        self.patient1 = Patient(doctor_id=self.doc1.id, first_name="John", last_name="Doe", age=45, gender="Male")
        db.session.add(self.patient1)
        db.session.commit()

        self.consult1 = Consultation(patient_id=self.patient1.id, last_question_tag="Fever")
        db.session.add(self.consult1)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def login(self, username, password):
        return self.client.post('/login', data=dict(username=username, password=password), follow_redirects=True)

    # --- AUTHENTICATION & DATABASE EDGE CASES ---
    def test_signup_existing_username(self):
        response = self.client.post('/signup', data=dict(
            username="dr_smith", password="newpassword", department="Cardiology"
        ), follow_redirects=True)
        self.assertIn(b'Username already exists', response.data)

    def test_login_invalid_credentials(self):
        response = self.login("dr_smith", "wrongpassword")
        self.assertIn(b'Invalid username or password', response.data)

    def test_unauthenticated_access_protection(self):
        response = self.client.get('/dashboard', follow_redirects=True)
        self.assertIn(b'login', response.data.lower())

    def test_api_new_patient_creation(self):
        self.login("dr_smith", "password123")
        response = self.client.post('/api/patient/new', json={
            "firstName": "Jane", "lastName": "Doe", "age": 30, "gender": "Female", "smokerStatus": True
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")

    def test_patient_404_not_found(self):
        self.login("dr_smith", "password123")
        response = self.client.get('/patient/999')
        self.assertEqual(response.status_code, 404)

    # --- ROW-LEVEL SECURITY ---
    def test_rls_diagnostic_protection(self):
        self.login("dr_jones", "password123")
        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "Coughing"})
        self.assertEqual(response.status_code, 403) # Doctor 2 blocked from Doctor 1's patient

    # --- ROUTE NLP INTEGRATION ---
    @patch('routes.predict_disease')
    @patch('routes.extract_symptoms_robust')
    def test_affirmative_state_mapping(self, mock_extract, mock_predict):
        self.login("dr_smith", "password123")
        mock_extract.return_value = ([], [], []) 
        mock_predict.return_value = ("Influenza", 0.75, ["Fever"])

        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "Yes it is quite high"})
        data = json.loads(response.data)
        self.assertIn('Fever', data['symptoms'])

    @patch('routes.predict_disease')
    @patch('routes.extract_symptoms_robust')
    def test_negative_state_mapping(self, mock_extract, mock_predict):
        self.login("dr_smith", "password123")
        mock_extract.return_value = ([], [], []) 
        mock_predict.return_value = ("Acute laryngitis", 0.65, ["Fever"])

        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "No I have not had one"})
        data = json.loads(response.data)
        self.assertIn('Fever', data['denied'])

    @patch('routes.predict_disease')
    @patch('routes.extract_symptoms_robust')
    def test_high_confidence_termination(self, mock_extract, mock_predict):
        self.login("dr_smith", "password123")
        self.consult1.extracted_symptoms = "['Voice_Hoarseness', 'Sore_throat']"
        db.session.commit()

        mock_extract.return_value = ([], [], [])
        mock_predict.return_value = ("Viral pharyngitis", 0.88, [])

        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "Continue."})
        data = json.loads(response.data)
        self.assertIn("AI recommendation complete", data['response'])

    @patch('routes.predict_disease')
    @patch('routes.extract_symptoms_robust')
    def test_red_flag_alert_trigger(self, mock_extract, mock_predict):
        self.login("dr_smith", "password123")
        mock_extract.return_value = (['Shortness_of_breath'], [], [])
        mock_predict.return_value = ("Pneumonia", 0.70, [])

        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "I can't breathe."})
        data = json.loads(response.data)
        self.assertTrue(data['has_red_flag'])

    @patch('routes.predict_disease')
    @patch('routes.extract_symptoms_robust')
    def test_diagnose_empty_message(self, mock_extract, mock_predict):
        self.login("dr_smith", "password123")
        self.consult1.extracted_symptoms = "[]"
        db.session.commit()
        
        mock_extract.return_value = ([], [], [])
        mock_predict.return_value = ("Waiting...", 0.0, [])

        response = self.client.post(f'/diagnose/{self.consult1.id}', json={"message": "   "})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("I'm listening", data['response'])

if __name__ == '__main__':
    unittest.main(verbosity=2)