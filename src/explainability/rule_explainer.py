from src.explainability.dsm_phq import DSM_PHQ_MAPPING

# 🔥 Extended Bilingual + Hinglish Lexicon (All major depressive symptoms)
# Covers English, Hindi, and Hinglish spellings / variants / emotional slang
MULTILINGUAL_PHRASES = {
    1: [  # Loss of interest / Anhedonia
        "lost interest", "bored", "no motivation", "nothing excites", "don't enjoy", "dont enjoy",
        "mann nahi lagta", "kuch karne ka mann nahi hai", "interest nahi hai", "bored lag raha hai",
        "maza nahi aa raha", "karne ka mann nahi", "dil nahi lagta", "passion chala gaya", "kuch accha nahi lag raha"
    ],
    2: [  # Depressed mood / sadness
        "hopeless", "empty", "depressed", "sad", "crying", "pointless", "down", "lonely", "worthless",
        "dukhi", "udass", "nirash", "udaas", "dukhi hoon", "hopeless feel ho raha hai",
        "acha nahi lag raha", "mann udaas hai", "mann dukhi hai", "ro raha hoon", "dil toota hua hai",
        "bahut bura lag raha hai", "feeling low", "feel down"
    ],
    3: [  # Sleep disturbance
        "can't sleep", "cant sleep", "insomnia", "awake all night", "no sleep", "oversleep", "restless sleep",
        "neend nahi aa rahi", "raat bhar jagta hoon", "raat bhar jaag raha hoon", "so nahi pa raha",
        "neend kam ho gayi", "sote waqt uth jaata hoon", "neend nahi lagti", "raat bhar neend nahi", "sahi se nahi so pa raha"
    ],
    4: [  # Fatigue / low energy
        "tired", "exhausted", "drained", "fatigued", "no energy", "can't get up", "low energy", "weak",
        "thakan lag rahi hai", "energy nahi hai", "kamjor mehsoos ho raha hai", "thak gaya hoon",
        "bahut kamjor lagta hai", "bilkul himmat nahi", "uthne ka mann nahi", "body thaki hui hai"
    ],
    5: [  # Appetite / weight change
        "no appetite", "overeating", "binge", "eating less", "lost appetite", "eating too much", "not eating",
        "bhukh nahi lag rahi", "bhook nahi lagti", "zyaada kha raha hoon", "kam kha raha hoon",
        "weight kam ho gaya", "weight badh gaya", "khaane ka mann nahi", "kuch khane ka mann nahi"
    ],
    6: [  # Worthlessness / guilt
        "useless", "worthless", "failure", "hate myself", "ashamed", "guilty", "guilt", "self hate", "disgusted with myself",
        "apne aap se nafrat", "main failure hoon", "bekaar lagta hai", "apne aap ko pasand nahi karta",
        "main kharab insan hoon", "mujhse kuch nahi hota", "main kuch nahi kar sakta", "khud se ghin aati hai"
    ],
    7: [  # Concentration / cognitive issues
        "can't focus", "cant focus", "can't concentrate", "forget", "mind blank", "distracted", "confused",
        "dhyan nahi lagta", "focus nahi ho raha", "yaad nahi reh raha", "mann kahin aur hai",
        "soch nahi pa raha", "dimag kaam nahi kar raha", "mind off", "dimag thak gaya hai"
    ],
    8: [  # Psychomotor agitation / restlessness
        "restless", "uneasy", "fidget", "slow", "sluggish", "pacing", "irritable", "anxious", "panic", "can't sit still",
        "bechain", "bechaini", "uneasy feel ho raha hai", "andar ghabrahat hai", "ghabrahat", "chinta", "ghabrah raha hoon",
        "slow feel kar raha hoon", "dil ghabra raha hai", "man ghabra raha hai", "andar se ajeeb lag raha hai"
    ],
    9: [  # Suicidal ideation / death thoughts
        "suicide", "end my life", "kill myself", "don't want to live", "i want to die", "want to end it",
        "marne ka mann", "jeene ka mann nahi", "zindagi bekaar hai", "apna jeevan samapt", "mar jana chahta hoon",
        "jeevan khatam karna", "mujhe mar jana hai", "jeevan se thak gaya hoon", "jeene ka koi matlab nahi"
    ]
}

def detect_symptoms(text: str):
    """Detect depressive symptoms from text using multilingual keyword matching."""
    t = text.lower()
    matched = []

    for sid, phrases in MULTILINGUAL_PHRASES.items():
        for phrase in phrases:
            if phrase in t:
                matched.append({
                    'symptom_id': sid,
                    'symptom_label': DSM_PHQ_MAPPING[sid]['phq_label'],
                    'phq_label': DSM_PHQ_MAPPING[sid]['phq_label'],
                    'phq9_question': DSM_PHQ_MAPPING[sid]['phq_label'],  # PHQ-9 question text
                    'dsm_criteria': DSM_PHQ_MAPPING[sid]['dsm_criteria'],
                    'keyword_found': phrase,
                    'matched_keyword': phrase  # Alias for compatibility
                })
                break

    return matched

def detect_temporal_symptoms(text: str, timestamp=None):
    """
    Detect temporal depression symptoms from text and posting time.
    
    Research Basis:
    - Cosma et al. 2023 (Time-Enriched): "Late-night posting (2-4 AM) correlates 
      with sleep disturbance (r=0.42, p<0.001). Temporal features improve F1 by 3-5%."
    
    Temporal Indicators:
    1. Late-night posting (2-4 AM) → Sleep disturbance (DSM-5 criterion 3)
    2. Temporal symptom keywords → Sleep, time perception, isolation
    3. Weekend posting → Social isolation indicator
    
    Args:
        text: Input text
        timestamp: Optional datetime object for posting time
    
    Returns:
        Dictionary with temporal symptoms:
            - temporal_score: float (0-1)
            - temporal_symptoms: list of detected symptoms
            - temporal_explanation: str
    """
    from src.data.loaders import extract_temporal_features
    
    # Extract temporal features
    temporal_features = extract_temporal_features(text, timestamp)
    
    temporal_symptoms = []
    temporal_score = 0.0
    explanations = []
    
    # 1. Late-night posting → sleep disturbance
    if temporal_features['late_night_post']:
        temporal_score += 0.4
        temporal_symptoms.append({
            'symptom_id': 3,
            'phq_label': DSM_PHQ_MAPPING[3]['phq_label'],
            'dsm_criteria': DSM_PHQ_MAPPING[3]['dsm_criteria'],
            'evidence': f"Late-night posting at {temporal_features['hour']}:00 AM"
        })
        explanations.append(f"Posted at {temporal_features['hour']}:00 AM (sleep disturbance indicator)")
    
    # 2. Temporal symptom keywords
    if temporal_features['temporal_symptom_count'] > 0:
        temporal_score += min(0.3, temporal_features['temporal_symptom_count'] * 0.1)
        keyword_text = ', '.join(temporal_features['temporal_keywords'][:3])
        temporal_symptoms.append({
            'symptom_id': 3,
            'phq_label': DSM_PHQ_MAPPING[3]['phq_label'],
            'dsm_criteria': DSM_PHQ_MAPPING[3]['dsm_criteria'],
            'evidence': f"Temporal keywords: {keyword_text}"
        })
        explanations.append(f"Detected {temporal_features['temporal_symptom_count']} temporal symptom keywords")
    
    # 3. Weekend posting → social isolation
    if temporal_features['weekend_post']:
        temporal_score += 0.2
        explanations.append(f"Posted on {temporal_features['day_of_week']} (possible social isolation)")
    
    # Cap score at 1.0
    temporal_score = min(1.0, temporal_score)
    
    return {
        'temporal_score': temporal_score,
        'temporal_symptoms': temporal_symptoms,
        'temporal_explanation': ' | '.join(explanations) if explanations else 'No temporal indicators detected',
        'temporal_features': temporal_features
    }

def explain_prediction(text: str):
    """
    Explain depression prediction based on detected symptoms.
    Returns format compatible with SafetyGuard and scripts.
    """
    ms = detect_symptoms(text)
    if not ms:
        return {
            'prediction': 'No depressive cues detected',
            'explanation': [],
            'symptom_count': 0,
            'detected_symptoms': [],
            'severity': 'none'
        }
    
    expl = [f"Detected '{m['keyword_found']}' → {m['phq_label']} ({m['dsm_criteria']})" for m in ms]
    
    # Determine severity based on symptom count
    if len(ms) == 1:
        sev = 'mild'
    elif len(ms) <= 3:
        sev = 'moderate'
    else:
        sev = 'severe'
    
    return {
        'prediction': f'{sev.title()} depressive cues',
        'explanation': ' | '.join(expl),
        'symptom_count': len(ms),
        'detected_symptoms': ms,
        'severity': sev
    }

