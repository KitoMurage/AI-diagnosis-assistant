from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models import Doctor, Patient, Consultation
from utils import extract_symptoms_robust, get_next_question, predict_disease
from datetime import datetime
import ast

main = Blueprint('main', __name__)

# --- 0. TRAFFIC CONTROLLER (The Missing Route!) ---
@main.route('/')
def home():
    """
    If someone goes to the base URL, redirect them.
    If they are logged in, go to the dashboard.
    If they are logged out, force them to the login screen.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.login'))

# --- 1. SYSTEM SETUP & AUTHENTICATION ---

@main.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        department = request.form.get('department')
        
        # Validation: Check if username already exists
        user_exists = Doctor.query.filter_by(username=username).first()
        if user_exists:
            flash("Username already exists. Please choose another one.", "error")
            return redirect(url_for('main.signup'))
            
        # Create new doctor
        new_doc = Doctor(username=username, department=department)
        new_doc.set_password(password)
        db.session.add(new_doc)
        db.session.commit()
        
        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for('main.login'))
        
    return render_template('signup.html')

@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
        
    if request.method == 'POST':
        # Using request.form for standard HTML form submissions
        username = request.form.get('username')
        password = request.form.get('password')
        
        doctor = Doctor.query.filter_by(username=username).first()
        
        # Verify user exists AND password is correct
        if doctor and doctor.check_password(password):
            login_user(doctor)
            return redirect(url_for('main.dashboard'))
            
        # If it fails, flash an error and reload the page
        flash("Invalid username or password. Please try again.", "error")
        return redirect(url_for('main.login'))
        
    return render_template('login.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been securely logged out.", "success")
    return redirect(url_for('main.login'))


# --- 2. PROTECTED CLINICAL ROUTES ---

@main.route('/dashboard')
@login_required
def dashboard():
    """
    ROW-LEVEL SECURITY: The database only returns patients where doctor_id matches 
    the currently logged-in doctor. It is impossible to see other doctors' patients.
    """
    my_patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.id.desc()).all()
    
    processed_patients = []
    for p in my_patients:
        # Get their most recent consultation
        last_consult = Consultation.query.filter_by(patient_id=p.id).order_by(Consultation.timestamp.desc()).first()
        
        processed_patients.append({
            'id': p.id,
            'name': f"{p.first_name} {p.last_name}",
            'age': p.age,
            'gender': p.gender,
            'diagnosis': last_consult.diagnosis if last_consult else "No Data",
            'confidence': last_consult.confidence if last_consult else 0.0
        })
        
    return render_template('dashboard.html', patients=processed_patients, doctor_name=current_user.username)

@main.route('/api/patient/new', methods=['POST'])
@login_required
def new_patient():
    """Receives data from the intake form and creates a new Patient record."""
    data = request.json
    
    patient = Patient(
        doctor_id=current_user.id,
        first_name=data.get('firstName'),
        last_name=data.get('lastName'),
        age=int(data.get('age')),
        gender=data.get('gender'),
        smoker_status=data.get('smokerStatus', False)
    )
    db.session.add(patient)
    db.session.commit()
    
    # Start a blank consultation for them instantly
    consult = Consultation(patient_id=patient.id)
    db.session.add(consult)
    db.session.commit()
    
    return jsonify({"status": "success", "consult_id": consult.id})

# --- 3. THE AI INFERENCE ENGINE ---
# Notice we now pass consult_id in the URL instead of using an anonymous session!
@main.route('/diagnose/<int:consult_id>', methods=['POST'])
@login_required
def diagnose(consult_id):
    user_text = request.json.get('message', '').strip()
    
    # Load the specific consultation
    record = Consultation.query.get_or_404(consult_id)
    
    # Security check: Ensure this consultation belongs to a patient of THIS doctor
    if record.patient.doctor_id != current_user.id:
        return jsonify({"error": "Unauthorized Access"}), 403

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
    # (In the future, you could pass record.patient.age into predict_disease here!)
    top_disease, confidence = predict_disease(current_symptoms)
    
    record.diagnosis = top_disease
    record.confidence = confidence
    
    # 4. Generate Response
    bot_response = ""
    next_tag = None

    if not current_symptoms:
        bot_response = "I'm listening. Please describe the patient's symptoms."
    elif confidence > 0.85:
        bot_response = f"Current Analysis: {top_disease} ({confidence*100:.1f}%).\nConfidence is high. AI recommendation complete."
    else:
        question_text, next_tag = get_next_question(current_symptoms, denied_symptoms)
        
        if new_pending:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n(Tracking Question about {new_pending[0]})"
            next_tag = new_pending[0]
        else:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n\nSuggested Next Question: {question_text}"

    if next_tag: record.last_question_tag = next_tag
        
    record.timestamp = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        "response": bot_response,
        "diagnosis": record.diagnosis,
        "confidence": f"{record.confidence*100:.1f}%",
        "symptoms": current_symptoms,
        "denied": denied_symptoms
    })

# We need a route to serve the actual recording UI for a specific patient
@main.route('/session/<int:consult_id>')
@login_required
def session_view(consult_id):
    consult = Consultation.query.get_or_404(consult_id)
    if consult.patient.doctor_id != current_user.id:
        return "Unauthorized", 403
    return render_template('index.html', consult_id=consult_id, patient=consult.patient)