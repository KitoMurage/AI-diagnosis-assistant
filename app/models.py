from extensions import db
from datetime import datetime

class PatientSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True)
    extracted_symptoms = db.Column(db.String(1000), default="[]") 
    denied_symptoms = db.Column(db.String(1000), default="[]") 
    last_question_tag = db.Column(db.String(100), nullable=True)
    diagnosis = db.Column(db.String(100))
    confidence = db.Column(db.Float, default=0.0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)