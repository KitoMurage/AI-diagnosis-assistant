import pandas as pd
import ast
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import TARGET_DISEASES, CODE_MAP  # <-- Importing your central config

print("🚀 Starting Neural Network Training...")

# 1. Load Data
df = pd.read_csv('dataset/train.csv', usecols=['PATHOLOGY', 'EVIDENCES'])
df = df[df['PATHOLOGY'].isin(TARGET_DISEASES)].copy()

# 2. Translate Codes
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
X = X.T.groupby(level=0).max().T # Deduplicate columns

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Neural Network
print("🧠 Training Neural Network (MLP)...")
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)

# 5. Evaluate
print("\n--- 📊 Neural Network Results ---")
y_pred = nn_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(nn_model, 'dataset/nn_model.pkl')
print("\n💾 Saved nn_model.pkl")