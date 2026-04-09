from flask import Blueprint, render_template, request, jsonify, session
from extensions import db
from models import PatientSession
from utils import extract_symptoms_robust, get_next_question, predict_disease
from datetime import datetime
import uuid
import ast

main = Blueprint('main', __name__)

@main.route('/')
def home():
    if 'uid' not in session: 
        session['uid'] = str(uuid.uuid4())
    return render_template('index.html')

@main.route('/reset', methods=['POST'])
def reset():
    if 'uid' in session: 
        session['uid'] = str(uuid.uuid4())
    return jsonify({"status": "cleared"})

@main.route('/dashboard')
def dashboard():
    """
    Clean MVC Controller for the Dashboard.
    Fetches database records, formats them safely, and passes them to the View.
    """
    all_patients = PatientSession.query.order_by(PatientSession.updated_at.desc()).all()
    
    # Process data cleanly in Python before sending it to HTML
    processed_patients = []
    for p in all_patients:
        try: 
            symptoms = ast.literal_eval(p.extracted_symptoms)
        except: 
            symptoms = []
            
        try: 
            denied = ast.literal_eval(p.denied_symptoms)
        except: 
            denied = []
            
        processed_patients.append({
            'id': p.session_id[:8],
            'time': p.updated_at.strftime('%H:%M:%S'),
            'diagnosis': p.diagnosis,
            'confidence': p.confidence,
            'symptoms': symptoms,
            'denied': denied
        })
        
    # Send processed data to a dedicated HTML file
    return render_template('dashboard.html', patients=processed_patients)

@main.route('/diagnose', methods=['POST'])
def diagnose():
    user_text = request.json.get('message', '').strip()
    uid = session.get('uid')
    
    record = PatientSession.query.filter_by(session_id=uid).first()
    if not record:
        record = PatientSession(session_id=uid)
        db.session.add(record)
    
    # Load Memory
    current_symptoms = ast.literal_eval(record.extracted_symptoms) if record.extracted_symptoms else []
    denied_symptoms = ast.literal_eval(record.denied_symptoms) if record.denied_symptoms else []
    
    # 1. Extract Logic
    new_confirmed, new_denied, new_pending = extract_symptoms_robust(user_text)
    
    for s in new_confirmed:
        if s not in current_symptoms and s not in denied_symptoms: current_symptoms.append(s)
    for s in new_denied:
        if s not in denied_symptoms and s not in current_symptoms: denied_symptoms.append(s)

    # 2. Cross-Turn Logic
    affirmations = ['yes', 'yeah', 'yep', 'correct', 'i do', 'sure']
    negations = ['no', 'nope', 'not', "don't", 'dont', 'never']
    
    is_affirmative = any(w in user_text.lower().split() for w in affirmations)
    is_negative = any(w in user_text.lower().split() for w in negations)

    if record.last_question_tag:
        if is_affirmative and record.last_question_tag not in current_symptoms:
            current_symptoms.append(record.last_question_tag)
        elif is_negative and record.last_question_tag not in denied_symptoms:
            denied_symptoms.append(record.last_question_tag)

    if new_pending:
        record.last_question_tag = new_pending[0]
    else:
        record.last_question_tag = None

    record.extracted_symptoms = str(current_symptoms)
    record.denied_symptoms = str(denied_symptoms)
    
    # 3. Predict
    top_disease, confidence = predict_disease(current_symptoms)
    
    record.diagnosis = top_disease
    record.confidence = confidence
    
    # 4. Generate Response
    bot_response = ""
    next_tag = None

    if not current_symptoms:
        bot_response = "I'm listening. Please describe your symptoms."
    elif confidence > 0.85:
        bot_response = f"Current Analysis: {top_disease} ({confidence*100:.1f}%).\nConfidence is high. AI recommendation complete."
    else:
        question_text, next_tag = get_next_question(current_symptoms, denied_symptoms)
        
        if new_pending:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n(Tracking Doctor's Question about {new_pending[0]})"
            next_tag = new_pending[0]
        else:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n\nSuggested Next Question: {question_text}"

    if next_tag: record.last_question_tag = next_tag
        
    record.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        "response": bot_response,
        "diagnosis": record.diagnosis,
        "confidence": f"{record.confidence*100:.1f}%",
        "symptoms": current_symptoms,
        "denied": denied_symptoms
    })