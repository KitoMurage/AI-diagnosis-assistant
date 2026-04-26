import pandas as pd
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from config import TARGET_DISEASES, CODE_MAP

print("🔍 Loading dataset and preprocessing to match training environment...")

try:
    df = pd.read_csv('ml_pipeline/dataset/train.csv', usecols=['PATHOLOGY', 'EVIDENCES'])
except FileNotFoundError:
   
    df = pd.read_csv('dataset/train.csv', usecols=['PATHOLOGY', 'EVIDENCES'])

df = df[df['PATHOLOGY'].isin(TARGET_DISEASES)].copy()

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

print("✅ Data ready. Loading saved models...")
try:
    rf_model = joblib.load('ml_pipeline/dataset/rf_model.pkl')
    nn_model = joblib.load('ml_pipeline/dataset/nn_model.pkl')
except FileNotFoundError:
    rf_model = joblib.load('dataset/rf_model.pkl')
    nn_model = joblib.load('dataset/nn_model.pkl')

# RANDOM FOREST EVALUATION
print("\n==================================================")
print("🧠 RANDOM FOREST METRICS")
print("==================================================")
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {rf_acc * 100:.2f}%")
print(f"Precision: {rf_prec * 100:.2f}%")
print(f"Recall:    {rf_rec * 100:.2f}%")

print("\n📉 RANDOM FOREST CONFUSION MATRIX")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, rf_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
print(cm_df)


# NEURAL NETWORK EVALUATION
print("\n==================================================")
print("🤖 NEURAL NETWORK METRICS")
print("==================================================")
nn_pred = nn_model.predict(X_test)

nn_acc = accuracy_score(y_test, nn_pred)
nn_prec = precision_score(y_test, nn_pred, average='weighted', zero_division=0)
nn_rec = recall_score(y_test, nn_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {nn_acc * 100:.2f}%")
print(f"Precision: {nn_prec * 100:.2f}%")
print(f"Recall:    {nn_rec * 100:.2f}%")