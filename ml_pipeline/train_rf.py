import pandas as pd
import ast
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import TARGET_DISEASES, CODE_MAP  # <-- This is the magic line!

print("🚀 Starting Random Forest Training...")

df = pd.read_csv('dataset/train.csv', usecols=['PATHOLOGY', 'EVIDENCES'])
df = df[df['PATHOLOGY'].isin(TARGET_DISEASES)].copy()

print("⚙️ Translating Codes to English...")
rows_list = []
for _, row in df.iterrows():
    evidences = ast.literal_eval(row['EVIDENCES']) 
    patient_vector = {'pathology': row['PATHOLOGY']}
    for code in evidences:
        if code in CODE_MAP:
            patient_vector[CODE_MAP[code]] = 1
        elif code.split('_@_')[0] in CODE_MAP:
            base = code.split('_@_')[0]
            patient_vector[CODE_MAP[base]] = 1
    rows_list.append(patient_vector)

df_clean = pd.DataFrame(rows_list).fillna(0)
X = df_clean.drop(columns=['pathology'])
y = df_clean['pathology']
X = X.T.groupby(level=0).max().T 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

print("\n--- 📊 Random Forest Results ---")
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n--- 🔍 Top 5 Important Symptoms ---")
importances = rf_model.feature_importances_
feature_df = pd.DataFrame({'Symptom': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_df.head(5))

joblib.dump(rf_model, 'dataset/rf_model.pkl')
joblib.dump(list(X.columns), 'dataset/symptom_list.pkl')
print("\n💾 Saved rf_model.pkl and symptom_list.pkl")