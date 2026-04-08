# AI-diagnosis-assistant/ml_pipeline/config.py

TARGET_DISEASES = [
    'Viral pharyngitis', 'Acute laryngitis', 'Acute rhinosinusitis', 
    'Influenza', 'Pneumonia', 'Acute bronchitis', 
    'Chronic obstructive pulmonary disease', 'Asthma'
]

CODE_MAP = {
    'E_91': 'Fever', 'E_201': 'Cough', 'E_97': 'Sore_throat', 
    'E_66': 'Shortness_of_breath', 'E_94': 'Chills', 'E_88': 'Fatigue_Severe',
    'E_50': 'Sweating_Increased', 'E_45': 'Coughing_Blood', 'E_144': 'Pain_Muscle_Diffuse',
    'E_161': 'Loss_Appetite', 'E_103': 'Loss_of_Smell', 'E_212': 'Voice_Hoarseness',
    'E_219': 'Symptoms_Worse_At_Night', 'E_220': 'Pain_Breathing_Deeply',
    'E_181': 'Runny_Nose_Clear', 'E_182': 'Nasal_Discharge_Green_Yellow',
    'E_77': 'Sputum_Colored', 'E_120': 'Nasal_Polyps', 'E_121': 'Deviated_Septum',
    'E_55_@_V_89': 'Pain_Forehead', 'E_55_@_V_166': 'Pain_Temple_R',
    'E_55_@_V_167': 'Pain_Temple_L', 'E_55_@_V_108': 'Pain_Cheek_R',
    'E_55_@_V_109': 'Pain_Cheek_L', 'E_55_@_V_122': 'Pain_Nose',
    'E_55_@_V_124': 'Pain_Back_of_Head', 'E_55_@_V_125': 'Pain_Eye_R',
    'E_55_@_V_126': 'Pain_Eye_L', 'E_55_@_V_174': 'Pain_Trachea',
    'E_55_@_V_148': 'Pain_Pharynx', 'E_55_@_V_20': 'Pain_Tonsil_R',
    'E_55_@_V_21': 'Pain_Tonsil_L', 'E_55_@_V_33': 'Pain_Thyroid',
    'E_55_@_V_53': 'Pain_Neck_Side_R', 'E_55_@_V_54': 'Pain_Neck_Side_L',
    'E_55_@_V_26': 'Pain_Neck_Back', 'E_55_@_V_163': 'Pain_Jaw',
    'E_55_@_V_101': 'Pain_Chest_Upper', 'E_55_@_V_29': 'Pain_Chest_Lower',
    'E_55_@_V_55': 'Pain_Chest_Side_R', 'E_55_@_V_56': 'Pain_Chest_Side_L',
    'E_55_@_V_170': 'Pain_Chest_Back_R', 'E_55_@_V_171': 'Pain_Chest_Back_L',
    'E_55_@_V_159': 'Pain_Breast_R', 'E_55_@_V_160': 'Pain_Breast_L',
    'E_129': 'Skin_Lesions', 'E_130_@_V_156': 'Rash_Color_Pink',
    'E_131_@_V_12': 'Rash_Peeling', 'E_132_@_0': 'Rash_Swollen_No',
    'E_132_@_1': 'Rash_Swollen_Yes', 'E_135_@_V_12': 'Lesion_Large_gt_1cm',
    'E_79': 'History_Smoker', 'E_78': 'History_Alcohol', 'E_123': 'History_COPD',
    'E_124': 'History_Asthma', 'E_125': 'History_GERD', 'E_118': 'History_Pneumonia',
    'E_106': 'History_Heart_Failure', 'E_107': 'History_Stroke', 'E_196': 'History_Surgery_Recent',
    'E_227': 'Risk_Immunosuppressed', 'E_226': 'Risk_Allergies', 'E_116': 'Risk_Recent_Cold',
    'E_41': 'Risk_Contact_Sick_Person', 'E_48': 'Risk_Crowded_Housing', 'E_49': 'Risk_Daycare',
    'E_204_@_V_4': 'Risk_Travel_NorthAmerica', 'E_54_@_V_181': 'Pain_Char_Burning',
    'E_54_@_V_192': 'Pain_Char_Sharp', 'E_54_@_V_183': 'Pain_Char_Heavy',
    'E_54_@_V_179': 'Pain_Char_Knife', 'E_54_@_V_161': 'Pain_Char_Sensitive'
}