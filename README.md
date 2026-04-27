# Ambient Clinical Copilot: Respiratory Diagnostic Support System

An intelligent, voice-enabled assistant designed to reduce the manual data-entry burden for clinicians while providing real-time diagnostic decision support using the DDXPlus dataset.

## 🚀 System Architecture
The system follows a **3-tier architecture**:
1.  **Frontend (Client):** JavaScript-driven interface utilizing the **Web Speech API** for ambient transcription.
2.  **Backend (Server):** **Flask** application managing session state, authentication, and database ORM via **SQLAlchemy**.
3.  **Diagnostic Engine (ML):** A **Random Forest Classifier** trained on filtered respiratory data from the **DDXPlus** dataset, integrated with a custom **Information Gain Router** for dynamic questioning.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9 or higher
* Google Chrome (required for Web Speech API support)

### Setup Instructions
1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd respiratory-copilot
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

4.  **Database Initialization:**
    The system uses SQLite. To initialize the schema:
    ```bash
    python -c "from app import db; db.create_all()"
    ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    Access the application at `http://127.0.0.1:5000`.

## 📖 User Manual

### 1. Clinician Authentication
Log in with your credentials. The system uses **Werkzeug** for secure password hashing. 

### 2. Patient Roster
Select a patient. The system enforces **Row-Level Security (RLS)**, ensuring clinicians only access patients assigned to their specific `doctor_id`.

### 3. Ambient Consultation
* **Start Listening:** Click the microphone icon to begin the transcription.
* **Natural Conversation:** Speak naturally with the patient. The **NLP State Machine** will extract symptoms in real-time.
* **Negation Handling:** If you ask "Do you have a cough?" and the patient says "No," the system identifies this as a negated symptom and updates the "Ruled Out" array.

### 4. Diagnostic Feedback
* **Probability Display:** The sidebar updates with the top 3 most likely respiratory conditions.
* **Suggested Questions:** The system uses **Information Gain** to suggest the next question that will most effectively narrow down the differential.

## 🛡️ Safety & Security
* **Clinical Red Flags:** A heuristic matrix monitors for "Red Flag" symptoms (e.g., severe chest pain). If detected, a high-priority UI alert overrides ML predictions to ensure immediate triage.
* **Data Integrity:** All interactions are handled via **SQLAlchemy ORM** to prevent SQL Injection attacks.

## 📊 Dataset Attribution
This project utilizes a filtered subset of the **DDXPlus Dataset** (Mila - Quebec AI Institute). Original alphanumeric disease codes have been mapped to clinical respiratory terms for user readability.

---
*Developed as part of a University Research Project.*
