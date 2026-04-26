from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db
from models import Doctor, Patient, Consultation
from utils import extract_symptoms_robust, get_next_question, predict_disease
from datetime import datetime
import ast

main = Blueprint('main', __name__)


@main.route('/')
def home():
    """
    If someone goes to the base URL, redirect them.
    If logged in go to the dashboard.
    If logged out force the login screen.
    """
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('main.login'))

# SYSTEM SETUP & AUTHENTICATION

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
        
        # Verify user exists and password is correct
        if doctor and doctor.check_password(password):
            login_user(doctor)
            return redirect(url_for('main.dashboard'))
            
        # flash an error and reload the page
        flash("Invalid username or password. Please try again.", "error")
        return redirect(url_for('main.login'))
        
    return render_template('login.html')

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been securely logged out.", "success")
    return redirect(url_for('main.login'))


# PROTECTED CLINICAL ROUTES

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

# --- ADMIN ANALYTICS DASHBOARD ---
@main.route('/admin')
@login_required
def admin_dashboard():
    # Security: Kick out anyone who isn't the admin
    if current_user.username != 'admin':
        flash("Unauthorized access. Admin privileges required.", "error")
        return redirect(url_for('main.dashboard'))
        
    # Aggregate Data Across ALL Doctors
    all_patients = Patient.query.all()
    all_consults = Consultation.query.filter(Consultation.diagnosis != "Pending").all()
    
    total_patients = len(all_patients)
    total_consults = len(all_consults)
    
    # Calculate Average AI Confidence
    avg_confidence = 0
    if total_consults > 0:
        total_conf = sum(c.confidence for c in all_consults)
        avg_confidence = (total_conf / total_consults) * 100
        
    # Epidemiological Tracking (Count diseases)
    disease_counts = {}
    for c in all_consults:
        disease = c.diagnosis
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
    # Sort for the chart
    sorted_diseases = dict(sorted(disease_counts.items(), key=lambda item: item[1], reverse=True))

    return render_template('admin_dashboard.html', 
                           total_patients=total_patients,
                           total_consults=total_consults,
                           avg_confidence=f"{avg_confidence:.1f}%",
                           disease_labels=list(sorted_diseases.keys()),
                           disease_data=list(sorted_diseases.values()))

@main.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    """Detailed EHR view for a specific patient."""
    patient = Patient.query.get_or_404(patient_id)
    
    # Security: Ensure this patient belongs to the logged-in doctor
    if patient.doctor_id != current_user.id:
        flash("Unauthorized access. This patient is not in your roster.", "error")
        return redirect(url_for('main.dashboard'))

    # Fetch all consultations for this patient, newest first
    consults = Consultation.query.filter_by(patient_id=patient.id).order_by(Consultation.timestamp.desc()).all()
    
    processed_consults = []
    for c in consults:
        # Convert the stringified database lists back into actual Python lists
        symptoms = ast.literal_eval(c.extracted_symptoms) if c.extracted_symptoms else []
        denied = ast.literal_eval(c.denied_symptoms) if c.denied_symptoms else []
        
        # Dynamically check for red flags in this specific historical consultation
        red_flag_watchlist = ['Shortness_of_breath', 'Coughing_Blood', 'Pain_Chest_Upper', 'Wheezing', 'Cyanosis', 'Altered_Mental_Status', 'Stridor', 'Tachycardia']
        triggered_flags = [sym for sym in symptoms if sym in red_flag_watchlist]

        processed_consults.append({
            'id': c.id,
            'timestamp': c.timestamp.strftime("%b %d, %Y - %H:%M"),
            'diagnosis': c.diagnosis,
            'confidence': f"{c.confidence * 100:.1f}%" if c.confidence > 0 else "Pending",
            'symptoms': [s.replace('_', ' ') for s in symptoms],
            'denied': [s.replace('_', ' ') for s in denied],
            'red_flags': [s.replace('_', ' ') for s in triggered_flags],
            'transcript': c.transcript if c.transcript else "No transcript recorded."
        })

    return render_template('patient_detail.html', patient=patient, consults=processed_consults)

# --- 3. THE AI INFERENCE ENGINE ---
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

    # 2. Cross-Turn Logic (Strictly obeys the DOCTOR's last asked question)
    affirmations = ['yes', 'yeah', 'yep', 'correct', 'i do', 'sure']
    negations = ['no', 'nope', 'not', "don't", 'dont', 'never']
    
    is_affirmative = any(w in user_text.lower().split() for w in affirmations)
    is_negative = any(w in user_text.lower().split() for w in negations)

    if record.last_question_tag:
        if is_affirmative and record.last_question_tag not in current_symptoms:
            current_symptoms.append(record.last_question_tag)
        elif is_negative and record.last_question_tag not in denied_symptoms:
            denied_symptoms.append(record.last_question_tag)

    # Update state: Only remember if the DOCTOR just asked a new question
    if new_pending:
        record.last_question_tag = new_pending[0]
    else:
        record.last_question_tag = None

    record.extracted_symptoms = str(current_symptoms)
    record.denied_symptoms = str(denied_symptoms)
    
    # 3. Predict 
    top_disease, confidence, xai_factors = predict_disease(current_symptoms)
    
    record.diagnosis = top_disease
    record.confidence = confidence
    
    # Check for Red Flags
    has_red_flag = any(sym in current_symptoms for sym in ['Shortness_of_breath', 'Coughing_Blood', 'Pain_Chest_Upper', 'Wheezing'])
    
    # 4. Generate Response
    bot_response = ""

    if not current_symptoms:
        bot_response = "I'm listening. Please describe the patient's symptoms."
    elif confidence > 0.85:
        bot_response = f"Current Analysis: {top_disease} ({confidence*100:.1f}%).\nConfidence is high. AI recommendation complete."
    else:
        question_text, _ = get_next_question(current_symptoms, denied_symptoms) # AI suggestion tag ignored for state memory
        
        if new_pending:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n(Tracking Question about {new_pending[0]})"
        else:
            bot_response = f"Current Analysis: {top_disease} ({confidence*100:.0f}%).\n\nSuggested Next Question: {question_text}"

    # NOTE: We no longer save the AI's suggestion to record.last_question_tag!
    # The system now only tracks actual spoken questions via new_pending above.

    current_transcript = record.transcript if record.transcript else ""
    record.transcript = current_transcript + f"Patient: {user_text}\nAI: {bot_response}\n\n"
        
    record.timestamp = datetime.utcnow()
    db.session.commit()
    
    return jsonify({
        "response": bot_response,
        "diagnosis": record.diagnosis,
        "confidence": f"{record.confidence*100:.1f}%",
        "symptoms": current_symptoms,
        "denied": denied_symptoms,
        "xai_factors": xai_factors,    
        "has_red_flag": has_red_flag    
    })

# We need a route to serve the actual recording UI for a specific patient
@main.route('/session/<int:consult_id>')
@login_required
def session_view(consult_id):
    consult = Consultation.query.get_or_404(consult_id)
    if consult.patient.doctor_id != current_user.id:
        return "Unauthorized", 403
    return render_template('index.html', consult_id=consult_id, patient=consult.patient)