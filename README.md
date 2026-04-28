# Ambient Clinical Copilot: Respiratory Diagnostic Support System

An intelligent, voice-enabled assistant designed to reduce the manual data-entry burden for clinicians while providing real-time diagnostic decision support using the DDXPlus dataset. 

## 🚀 System Architecture
The system follows a **3-tier architecture**:
1.  **Frontend (Client):** JavaScript-driven interface utilizing the **Web Speech API** for zero-latency, edge-computed ambient transcription.
2.  **Backend (Server):** **Flask** application managing session state, conversational context, and database ORM via **SQLAlchemy**.
3.  **Diagnostic Engine (ML):** A **Random Forest Classifier** trained on filtered respiratory data from the **DDXPlus** dataset, integrated with a custom **Clinical Rule-In Heuristic** for dynamic, differential questioning.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9 or higher
* Google Chrome (required for Web Speech API support)

### Setup Instructions
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/KitoMurage/AI-diagnosis-assistant.git](https://github.com/KitoMurage/AI-diagnosis-assistant.git)
    cd AI-diagnosis-assistant
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application:**
    *(Note: The SQLite database and tables will automatically initialize upon first run via the application factory).*
    ```bash
    python app/app.py
    ```
    Access the application at `http://127.0.0.1:5000`.

## 📖 User Manual & System Walkthrough

### 1. Clinician Authentication
Log in with your credentials. The system uses **Werkzeug** for secure password hashing to protect access to patient records.
> *[Replace with your image path]*
> `![Login Screen](assets/login.png)`

### 2. Clinician Dashboard & Patient Roster
View your active patient roster. The system enforces **Row-Level Security (RLS)**, ensuring clinicians can only access and manage patients strictly assigned to their specific `doctor_id`.
> *[Replace with your image path]*
> `![Patient Dashboard](assets/dashboard.png)`

### 3. Patient EHR & Consultation History
Access a detailed Electronic Health Record (EHR) for any specific patient. Review a chronological history of past consultations, complete with extracted symptoms, ruled-out symptoms, triggered red flags, and the raw archived conversation transcript.
> *[Replace with your image path]*
> `![Consultation History](assets/history.png)`

### 4. Ambient Consultation
* **Start Listening:** Click the microphone icon to begin edge-computed transcription.
* **Natural Conversation:** Speak naturally. The **NLP State Machine** will extract symptoms in real-time, completely ignoring conversational noise.
* **Negation Handling:** If you ask "Do you have a cough?" and the patient says "No," the system identifies this as a negated symptom and maps it to the "Ruled Out" array.
> *[Replace with your image path]*
> `![Live Consultation Interface](assets/consultation.png)`

### 5. Explainable AI & Diagnostic Feedback
* **Probability Display:** The sidebar updates dynamically with the top suspected respiratory conditions.
* **Diagnostic Rationale (XAI):** The system displays the exact mathematical weights (derived from Shannon Entropy) that drove the prediction, ensuring the model is not a "Black Box."
* **Clinical Rule-In Questioning:** Instead of static flowcharts, the system dynamically calculates and suggests the exact follow-up question that will most effectively confirm the leading differential diagnosis.
> *[Replace with your image path]*
> `![XAI and Diagnostic Feedback](assets/xai_feedback.png)`

### 6. Admin Analytics Dashboard
A secured, aggregated view of system-wide data (accessible only to users with Admin privileges). Tracks total patient volumes, average AI confidence scores, and generates real-time epidemiological charts of predicted respiratory diseases.
> *[Replace with your image path]*
> `![Admin Analytics Dashboard](assets/admin_dashboard.png)`

## 🛡️ Safety & Security
* **Clinical Red Flags:** A deterministic heuristic matrix monitors for critical "Red Flag" symptoms (e.g., severe chest pain, cyanosis). If detected, a high-priority UI alert overrides ML predictions to ensure immediate emergency triage.
* **Data Integrity:** All interactions are handled via **SQLAlchemy ORM** to prevent SQL Injection attacks. Raw transcripts are securely archived locally for clinical auditing.

## 📊 Dataset Attribution
This project utilizes a filtered subset of the **DDXPlus Dataset** (Mila - Quebec AI Institute). Original alphanumeric disease codes have been mapped to clinical respiratory terms for user readability.

---
*Developed as part of a University Research Project.*
