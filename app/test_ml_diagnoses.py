import unittest
from utils import extract_symptoms_robust, predict_disease

class TestOriginalModel(unittest.TestCase):

    def test_laryngitis_detection(self):
        """Case 1: Typical Laryngitis symptoms."""
        text = "My voice is very hoarse and I have a sore throat and a dry cough."
        confirmed, _, _ = extract_symptoms_robust(text)
        disease, confidence, _ = predict_disease(confirmed)
        print(f"\n[Laryngitis Test] Diagnosis: {disease} ({confidence*100:.1f}%)")
        self.assertEqual(disease, "Acute laryngitis")

    def test_pneumonia_detection(self):
        """Case 2: Typical Pneumonia symptoms (The 'Heavy' Test)."""
        text = "I have a high fever, chills, and I'm coughing up green sputum. My chest hurts when I breathe."
        confirmed, _, _ = extract_symptoms_robust(text)
        disease, confidence, _ = predict_disease(confirmed)
        print(f"\n[Pneumonia Test] Diagnosis: {disease} ({confidence*100:.1f}%)")
        # We want to see if the original model can actually land on Pneumonia
        self.assertEqual(disease, "Pneumonia")

    def test_influenza_detection(self):
        """Case 3: Typical Influenza symptoms."""
        text = "I feel exhausted, my muscles ache everywhere, I have a headache and a fever."
        confirmed, _, _ = extract_symptoms_robust(text)
        disease, confidence, _ = predict_disease(confirmed)
        print(f"\n[Influenza Test] Diagnosis: {disease} ({confidence*100:.1f}%)")
        self.assertEqual(disease, "Influenza")

    def test_sinusitis_detection(self):
        """Case 4: Typical Rhinosinusitis symptoms."""
        text = "My nose is blocked and I have a lot of pain in my forehead and cheeks."
        confirmed, _, _ = extract_symptoms_robust(text)
        disease, confidence, _ = predict_disease(confirmed)
        print(f"\n[Sinusitis Test] Diagnosis: {disease} ({confidence*100:.1f}%)")
        self.assertEqual(disease, "Acute rhinosinusitis")

if __name__ == '__main__':
    print("🧪 Testing Original ML Model Implementation...")
    unittest.main(verbosity=2)