"""
Core constants: DSM-5 and PHQ-9 symptom mappings for depression detection.

Based on clinical diagnostic criteria and validated against research literature.
"""
from typing import Dict, List

# DSM-5 Major Depressive Disorder Criteria (9 symptoms)
DSM5_SYMPTOMS = {
    1: {
        "label": "Anhedonia",
        "description": "Loss of interest or pleasure in activities",
        "phq9": "Little interest or pleasure in doing things",
        "keywords": [
            "lost interest", "don't enjoy", "no motivation", "nothing excites",
            "bored", "anhedonia", "no pleasure", "can't enjoy"
        ]
    },
    2: {
        "label": "Depressed Mood",
        "description": "Feeling sad, empty, or hopeless most of the day",
        "phq9": "Feeling down, depressed, or hopeless",
        "keywords": [
            "hopeless", "empty", "depressed", "sad", "crying", "pointless",
            "down", "lonely", "worthless", "despair", "miserable"
        ]
    },
    3: {
        "label": "Sleep Disturbance",
        "description": "Insomnia or hypersomnia nearly every day",
        "phq9": "Trouble falling or staying asleep, or sleeping too much",
        "keywords": [
            "can't sleep", "insomnia", "awake all night", "no sleep",
            "oversleep", "restless sleep", "sleep too much", "tired but can't sleep"
        ]
    },
    4: {
        "label": "Fatigue",
        "description": "Fatigue or loss of energy nearly every day",
        "phq9": "Feeling tired or having little energy",
        "keywords": [
            "tired", "exhausted", "drained", "fatigued", "no energy",
            "can't get up", "low energy", "weak", "lethargy"
        ]
    },
    5: {
        "label": "Appetite/Weight Change",
        "description": "Significant weight loss or gain, or change in appetite",
        "phq9": "Poor appetite or overeating",
        "keywords": [
            "no appetite", "overeating", "binge", "eating less", "lost appetite",
            "eating too much", "not eating", "weight loss", "weight gain"
        ]
    },
    6: {
        "label": "Worthlessness/Guilt",
        "description": "Feelings of worthlessness or excessive guilt",
        "phq9": "Feeling bad about yourself — or that you are a failure",
        "keywords": [
            "useless", "worthless", "failure", "hate myself", "ashamed",
            "guilty", "guilt", "self-hate", "disgusted with myself", "burden"
        ]
    },
    7: {
        "label": "Concentration Difficulty",
        "description": "Diminished ability to think or concentrate",
        "phq9": "Trouble concentrating on things",
        "keywords": [
            "can't focus", "can't concentrate", "forget", "mind blank",
            "distracted", "confused", "brain fog", "memory problems"
        ]
    },
    8: {
        "label": "Psychomotor Changes",
        "description": "Psychomotor agitation or retardation",
        "phq9": "Moving or speaking slowly, or being restless",
        "keywords": [
            "restless", "uneasy", "fidget", "slow", "sluggish",
            "pacing", "can't sit still", "agitated", "moving slowly"
        ]
    },
    9: {
        "label": "Suicidal Ideation",
        "description": "Recurrent thoughts of death or suicide",
        "phq9": "Thoughts that you would be better off dead or hurting yourself",
        "keywords": [
            "suicide", "end my life", "kill myself", "don't want to live",
            "want to die", "better off dead", "self-harm", "hurt myself"
        ]
    }
}

# Severity thresholds (PHQ-9 scoring)
SEVERITY_THRESHOLDS = {
    "none": (0, 0),
    "mild": (1, 4),
    "moderate": (5, 9),
    "moderately_severe": (10, 14),
    "severe": (15, 27)
}

# Multilingual keyword extensions (English + common social media variants)
EXTENDED_KEYWORDS = {
    1: ["nothing matters", "life is pointless", "don't care anymore", "numb"],
    2: ["feel like giving up", "want to disappear", "life is meaningless"],
    3: ["up all night", "sleep is impossible", "nightmare every night"],
    4: ["completely drained", "zero energy", "can barely move"],
    5: ["stopped eating", "eating everything", "food tastes like nothing"],
    6: ["everyone hates me", "i'm a terrible person", "ruin everything"],
    7: ["can't think straight", "brain doesn't work", "forgot my own name"],
    8: ["can't stop moving", "feel like i'm in slow motion"],
    9: ["planning to end it", "researching methods", "goodbye cruel world"]
}

# Crisis keywords (require immediate safety routing)
CRISIS_KEYWORDS = [
    "suicide plan", "going to kill myself", "tonight is my last",
    "writing goodbye notes", "collected pills", "found a method",
    "ready to end it", "final goodbye", "can't go on"
]

# Safety exclusions (phrases that indicate NOT at risk)
NEGATION_PATTERNS = [
    "not suicidal", "don't want to die", "would never hurt myself",
    "not depressed", "feeling better", "improved mood"
]


def get_symptom_keywords(symptom_id: int, include_extended: bool = True) -> List[str]:
    """Get all keywords for a specific DSM-5 symptom."""
    base = DSM5_SYMPTOMS.get(symptom_id, {}).get("keywords", [])
    if include_extended and symptom_id in EXTENDED_KEYWORDS:
        return base + EXTENDED_KEYWORDS[symptom_id]
    return base


def get_severity_level(symptom_count: int) -> str:
    """Map symptom count to PHQ-9 severity level."""
    for level, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= symptom_count <= high:
            return level
    return "unknown"


def is_crisis_text(text: str) -> bool:
    """Check if text contains crisis-level indicators."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)
