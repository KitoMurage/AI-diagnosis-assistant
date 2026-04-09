import pandas as pd
import json
import ast

# --- CONFIGURATION ---
TARGET_DISEASES = [
    'Viral pharyngitis', 'Acute laryngitis', 'Acute rhinosinusitis', 
    'Influenza', 'Pneumonia', 'Acute bronchitis', 
    'Chronic obstructive pulmonary disease', 'Asthma'
]

def get_codes():
    print("🚀 Scanning for Respiratory Symptom Codes...")
    
    # 1. Load Dictionary
    with open('dataset/release_evidences.json', 'r') as f:
        evidence_dict = json.load(f)

    # 2. Load Data
    try:
        df = pd.read_csv('dataset/train.csv', usecols=['PATHOLOGY', 'EVIDENCES'])
    except:
        print("❌ FATAL: dataset/train.csv missing.")
        return

    # 3. Filter
    df = df[df['PATHOLOGY'].isin(TARGET_DISEASES)]
    print(f"   Analyzing {len(df)} patient records...")

    # 4. Extract Unique Codes
    unique_codes = set()
    for _, row in df.iterrows():
        codes = ast.literal_eval(row['EVIDENCES'])
        unique_codes.update(codes)

    # 5. Generate Report
    print(f"\n✅ Found {len(unique_codes)} unique codes.")
    print("\n--- COPY EVERYTHING BELOW THIS LINE ---")
    print("Code,Raw_Name,Description_En")
    
    sorted_codes = sorted(list(unique_codes))
    
    for code in sorted_codes:
        # Handle complex codes like E_55_@_V_174
        if '_@_' in code:
            parts = code.split('_@_')
            base = parts[0]
            val = parts[1]
        else:
            base = code
            val = None
            
        if base in evidence_dict:
            meta = evidence_dict[base]
            raw_name = meta['name'] # e.g. "E_91" or "Fever"
            desc = meta.get('question_en', 'No description')
            
            # If there's a specific value meaning (e.g. Trachea)
            if val and 'value_meaning' in meta:
                val_map = meta['value_meaning']
                # Try finding the value mapping
                val_desc = ""
                if val in val_map:
                    val_desc = val_map[val].get('en', str(val_map[val]))
                elif val.replace('V_', '') in val_map:
                    clean_val = val.replace('V_', '')
                    val_desc = val_map[clean_val].get('en', str(val_map[clean_val]))
                
                if val_desc:
                    desc = f"{desc} [{val_desc}]"
            
            print(f"{code},{raw_name},{desc}")
        else:
            print(f"{code},UNKNOWN,Unknown Code")

    print("--- END OF COPY ---")

if __name__ == "__main__":
    get_codes()