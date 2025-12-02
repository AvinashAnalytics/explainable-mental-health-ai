"""
Clinical Validity Module: DSM-5 Criteria and PHQ-9 Score Estimation

Research Basis:
- DSM-5 (American Psychiatric Association, 2013)
- PHQ-9 (Kroenke et al., 2001): 9-item depression screening tool
- Clinical alignment per arXiv:2401.02984 requirements

DSM-5 Major Depressive Episode Criteria:
A. ≥5 symptoms during same 2-week period (must include 1 or 2):
   1. Depressed mood most of the day
   2. Markedly diminished interest/pleasure (anhedonia)
   3. Significant weight loss/gain or appetite change
   4. Insomnia or hypersomnia nearly every day  
   5. Psychomotor agitation or retardation
   6. Fatigue or loss of energy
   7. Feelings of worthlessness or excessive guilt
   8. Diminished ability to think/concentrate
   9. Recurrent thoughts of death/suicide

PHQ-9 Scoring:
- 0-4: Minimal depression
- 5-9: Mild depression
- 10-14: Moderate depression
- 15-19: Moderately severe depression
- 20-27: Severe depression
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


class DSM5Criteria:
    """
    DSM-5 Major Depressive Episode criteria checker.
    """
    
    # Complete DSM-5 symptom definitions with expanded keywords
    CRITERIA = {
        'A1_depressed_mood': {
            'name': 'Depressed mood most of the day',
            'keywords': [
                'depressed', 'sad', 'empty', 'hopeless', 'down', 'miserable',
                'tearful', 'crying', 'low mood', 'blue', 'unhappy', 'dejected',
                'melancholic', 'gloomy', 'despondent', 'despair', 'anguish'
            ],
            'patterns': [
                r'\bfeel\s+(so\s+)?(?:depressed|sad|empty|hopeless|down)',
                r'\b(?:everything|life)\s+(?:is|feels?)\s+(?:hopeless|empty|pointless)',
                r'\bcrying\s+(?:all|every)',
                r'\bcan\'t\s+stop\s+(?:crying|feeling\s+sad)'
            ],
            'weight': 3  # Core symptom
        },
        'A2_anhedonia': {
            'name': 'Markedly diminished interest or pleasure',
            'keywords': [
                'lost interest', 'no interest', 'don\'t enjoy', 'nothing matters',
                'anhedonia', 'apathetic', 'no pleasure', 'can\'t enjoy',
                'boring', 'meaningless', 'indifferent', 'numb', 'empty feeling',
                'don\'t care', 'pointless', 'no motivation', 'can\'t get excited'
            ],
            'patterns': [
                r'\blost\s+(?:all\s+)?interest',
                r'\bnothing\s+(?:brings|gives)\s+(?:me\s+)?(?:joy|pleasure|happiness)',
                r'\bdon\'t\s+(?:enjoy|care\s+about)\s+(?:anything|things)',
                r'\bused\s+to\s+(?:love|enjoy)\s+\w+\s+but\s+(?:now|don\'t)',
                r'\bcan\'t\s+(?:feel|find)\s+(?:joy|happiness|pleasure)'
            ],
            'weight': 3  # Core symptom
        },
        'A3_appetite_weight': {
            'name': 'Significant weight loss/gain or appetite change',
            'keywords': [
                'no appetite', 'not eating', 'lost appetite', 'can\'t eat',
                'overeating', 'binge', 'weight loss', 'weight gain',
                'eating too much', 'barely eating', 'food tastes like nothing',
                'force myself to eat', 'starving myself', 'comfort eating'
            ],
            'patterns': [
                r'\b(?:no|lost|barely\s+any)\s+appetite',
                r'\bcan\'t\s+(?:eat|stomach\s+food)',
                r'\bweight\s+(?:loss|gain)',
                r'\b(?:overeating|binge|eating\s+too\s+much)',
                r'\bfood\s+(?:tastes\s+like\s+nothing|doesn\'t\s+appeal)'
            ],
            'weight': 2
        },
        'A4_sleep_disturbance': {
            'name': 'Insomnia or hypersomnia nearly every day',
            'keywords': [
                'can\'t sleep', 'insomnia', 'no sleep', 'awake all night',
                'sleepless', 'tossing and turning', 'can\'t fall asleep',
                'wake up early', 'sleep all day', 'oversleep', 'sleeping too much',
                'can\'t get out of bed', 'exhausted but can\'t sleep',
                'sleep schedule ruined', 'up all night'
            ],
            'patterns': [
                r'\bcan\'t\s+(?:sleep|fall\s+asleep)',
                r'\b(?:awake|up)\s+all\s+night',
                r'\bsleep(?:ing)?\s+(?:all\s+day|too\s+much)',
                r'\bcan\'t\s+(?:get\s+)?(?:out\s+of\s+)?bed',
                r'\binsomnia'
            ],
            'weight': 2
        },
        'A5_psychomotor': {
            'name': 'Psychomotor agitation or retardation',
            'keywords': [
                'restless', 'fidgety', 'can\'t sit still', 'pacing',
                'slow', 'sluggish', 'moving in slow motion', 'lethargic',
                'heavy limbs', 'everything takes effort', 'can\'t get moving'
            ],
            'patterns': [
                r'\bcan\'t\s+sit\s+still',
                r'\bmoving\s+(?:so\s+)?slow',
                r'\beverything\s+(?:takes|requires)\s+(?:so\s+much\s+)?effort',
                r'\b(?:restless|fidgety|pacing)'
            ],
            'weight': 2
        },
        'A6_fatigue': {
            'name': 'Fatigue or loss of energy',
            'keywords': [
                'exhausted', 'no energy', 'tired', 'fatigued', 'drained',
                'burned out', 'can\'t get up', 'no strength', 'weak',
                'run down', 'wiped out', 'depleted', 'can barely function',
                'constantly tired', 'always exhausted'
            ],
            'patterns': [
                r'\b(?:no|zero|barely\s+any)\s+energy',
                r'\bso\s+(?:tired|exhausted|drained)',
                r'\bcan\'t\s+(?:get\s+up|function)',
                r'\b(?:always|constantly)\s+(?:tired|exhausted)',
                r'\bburned?\s+out'
            ],
            'weight': 2
        },
        'A7_worthlessness_guilt': {
            'name': 'Feelings of worthlessness or excessive guilt',
            'keywords': [
                'worthless', 'useless', 'failure', 'hate myself', 'burden',
                'good for nothing', 'ashamed', 'guilty', 'deserve this',
                'everyone would be better off', 'waste of space',
                'disappointed everyone', 'let everyone down', 'pathetic'
            ],
            'patterns': [
                r'\b(?:i\'m\s+)?(?:worthless|useless|a\s+failure)',
                r'\bhate\s+myself',
                r'\beveryone(?:\'s|\s+is)\s+better\s+off\s+without',
                r'\b(?:burden|waste)\s+(?:to|of)',
                r'\b(?:ashamed|guilty)\s+(?:of|about)'
            ],
            'weight': 2
        },
        'A8_concentration': {
            'name': 'Diminished ability to think or concentrate',
            'keywords': [
                'can\'t focus', 'can\'t concentrate', 'can\'t think',
                'brain fog', 'mind blank', 'distracted', 'forgetful',
                'can\'t make decisions', 'confused', 'spacey',
                'trouble focusing', 'can\'t remember', 'mental fog'
            ],
            'patterns': [
                r'\bcan\'t\s+(?:focus|concentrate|think)',
                r'\bbrain\s+fog',
                r'\bmind\s+(?:is\s+)?blank',
                r'\bcan\'t\s+(?:make\s+)?decisions',
                r'\b(?:trouble|difficulty)\s+(?:focusing|concentrating)'
            ],
            'weight': 2
        },
        'A9_suicidal_ideation': {
            'name': 'Recurrent thoughts of death or suicide',
            'keywords': [
                'suicide', 'suicidal', 'kill myself', 'end my life',
                'don\'t want to live', 'better off dead', 'no point living',
                'thoughts of death', 'ending it', 'self harm', 'hurt myself',
                'want to die', 'wish i was dead', 'can\'t go on'
            ],
            'patterns': [
                r'\b(?:suicide|suicidal|kill\s+myself)',
                r'\bend\s+(?:my|this)\s+life',
                r'\bdon\'t\s+want\s+to\s+(?:live|be\s+here)',
                r'\bbetter\s+off\s+dead',
                r'\bthoughts?\s+of\s+(?:death|suicide|dying)',
                r'\bwish\s+(?:i\s+)?(?:was|were)\s+dead'
            ],
            'weight': 5  # Critical symptom - immediate attention
        }
    }
    
    @classmethod
    def detect_symptoms(cls, text: str) -> Dict[str, Any]:
        """
        Detect DSM-5 symptoms in text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with:
                - detected_symptoms: List of detected criterion codes
                - symptom_details: Details for each symptom
                - meets_criteria: Whether DSM-5 criteria met (≥5 symptoms, including A1 or A2)
                - severity_estimate: Estimated severity based on symptom count
        """
        text_lower = text.lower()
        detected = {}
        
        for criterion_code, criterion_info in cls.CRITERIA.items():
            # Check keywords
            keyword_matches = []
            for keyword in criterion_info['keywords']:
                if keyword.lower() in text_lower:
                    keyword_matches.append(keyword)
            
            # Check regex patterns
            pattern_matches = []
            for pattern in criterion_info.get('patterns', []):
                if re.search(pattern, text_lower):
                    pattern_matches.append(pattern)
            
            # Symptom detected if any matches found
            if keyword_matches or pattern_matches:
                detected[criterion_code] = {
                    'name': criterion_info['name'],
                    'detected': True,
                    'keyword_matches': keyword_matches,
                    'pattern_matches': pattern_matches,
                    'weight': criterion_info['weight'],
                    'evidence': cls._extract_evidence(text, keyword_matches, pattern_matches)
                }
        
        # Check if DSM-5 criteria met
        num_symptoms = len(detected)
        has_core_symptom = 'A1_depressed_mood' in detected or 'A2_anhedonia' in detected
        meets_criteria = num_symptoms >= 5 and has_core_symptom
        
        # Severity estimation
        if num_symptoms == 0:
            severity = 'none'
        elif num_symptoms <= 2:
            severity = 'minimal'
        elif num_symptoms <= 4:
            severity = 'mild'
        elif num_symptoms <= 6:
            severity = 'moderate'
        elif num_symptoms <= 7:
            severity = 'moderately_severe'
        else:
            severity = 'severe'
        
        return {
            'detected_symptoms': list(detected.keys()),
            'symptom_details': detected,
            'num_symptoms': num_symptoms,
            'meets_dsm5_criteria': meets_criteria,
            'has_core_symptom': has_core_symptom,
            'severity_estimate': severity,
            'crisis_risk': 'A9_suicidal_ideation' in detected
        }
    
    @staticmethod
    def _extract_evidence(text: str, keyword_matches: List[str], pattern_matches: List[str]) -> str:
        """Extract text snippet showing evidence for symptom."""
        text_lower = text.lower()
        
        # Find first keyword match
        if keyword_matches:
            keyword = keyword_matches[0].lower()
            idx = text_lower.find(keyword)
            if idx != -1:
                # Extract surrounding context (±50 chars)
                start = max(0, idx - 50)
                end = min(len(text), idx + len(keyword) + 50)
                return text[start:end].strip()
        
        # Fallback: return first sentence
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()
        
        return text[:100]


class PHQ9Estimator:
    """
    PHQ-9 score estimation from text.
    
    Maps detected symptoms to PHQ-9 questionnaire items and estimates severity score.
    """
    
    # Mapping DSM-5 criteria to PHQ-9 items
    DSM5_TO_PHQ9 = {
        'A1_depressed_mood': 2,  # PHQ-9 item 2
        'A2_anhedonia': 1,  # PHQ-9 item 1
        'A4_sleep_disturbance': 3,  # PHQ-9 item 3
        'A6_fatigue': 4,  # PHQ-9 item 4
        'A3_appetite_weight': 5,  # PHQ-9 item 5
        'A7_worthlessness_guilt': 6,  # PHQ-9 item 6
        'A8_concentration': 7,  # PHQ-9 item 7
        'A5_psychomotor': 8,  # PHQ-9 item 8
        'A9_suicidal_ideation': 9  # PHQ-9 item 9
    }
    
    @classmethod
    def estimate_score(cls, dsm5_symptoms: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate PHQ-9 score from detected DSM-5 symptoms.
        
        Args:
            dsm5_symptoms: Output from DSM5Criteria.detect_symptoms()
        
        Returns:
            Dictionary with:
                - estimated_score: PHQ-9 score (0-27)
                - score_range: Range estimate
                - severity_level: Severity classification
                - item_scores: Per-item estimates
        """
        detected = dsm5_symptoms['symptom_details']
        item_scores = {}
        
        # Estimate score for each detected symptom
        for criterion_code, details in detected.items():
            if criterion_code in cls.DSM5_TO_PHQ9:
                phq9_item = cls.DSM5_TO_PHQ9[criterion_code]
                
                # Estimate severity (0-3) based on weight and evidence
                weight = details['weight']
                num_matches = len(details['keyword_matches']) + len(details['pattern_matches'])
                
                # Heuristic scoring
                if criterion_code == 'A9_suicidal_ideation':
                    score = 3  # Always severe if present
                elif weight >= 3 or num_matches >= 3:
                    score = 3  # Severe
                elif num_matches >= 2:
                    score = 2  # Moderate
                else:
                    score = 1  # Mild
                
                item_scores[phq9_item] = {
                    'score': score,
                    'symptom': criterion_code,
                    'evidence': details['evidence']
                }
        
        # Total score
        total_score = sum(item['score'] for item in item_scores.values())
        
        # Score range (accounting for uncertainty)
        score_lower = max(0, total_score - 2)
        score_upper = min(27, total_score + 2)
        
        # Severity classification
        if total_score <= 4:
            severity = 'minimal'
            severity_description = 'Minimal depression'
        elif total_score <= 9:
            severity = 'mild'
            severity_description = 'Mild depression'
        elif total_score <= 14:
            severity = 'moderate'
            severity_description = 'Moderate depression'
        elif total_score <= 19:
            severity = 'moderately_severe'
            severity_description = 'Moderately severe depression'
        else:
            severity = 'severe'
            severity_description = 'Severe depression'
        
        return {
            'estimated_score': total_score,
            'score_range': f'{score_lower}-{score_upper}',
            'severity_level': severity,
            'severity_description': severity_description,
            'item_scores': item_scores,
            'interpretation': cls._get_interpretation(total_score),
            'recommendation': cls._get_recommendation(total_score, dsm5_symptoms['crisis_risk'])
        }
    
    @staticmethod
    def _get_interpretation(score: int) -> str:
        """Get clinical interpretation of PHQ-9 score."""
        if score <= 4:
            return "Minimal or no depression. Monitor for changes."
        elif score <= 9:
            return "Mild depression. Consider watchful waiting and follow-up."
        elif score <= 14:
            return "Moderate depression. Treatment with counseling and/or medication may be beneficial."
        elif score <= 19:
            return "Moderately severe depression. Active treatment with counseling and medication recommended."
        else:
            return "Severe depression. Immediate treatment with medication and/or psychotherapy recommended. Consider referral to mental health specialist."
    
    @staticmethod
    def _get_recommendation(score: int, crisis_risk: bool) -> str:
        """Get clinical recommendation."""
        if crisis_risk:
            return "🚨 URGENT: Suicidal ideation detected. Immediate mental health evaluation required. Contact crisis services: 988 (US) or local emergency services."
        elif score >= 15:
            return "Strong recommendation: Consult mental health professional for comprehensive evaluation and treatment planning."
        elif score >= 10:
            return "Recommendation: Consider consultation with mental health professional or primary care physician."
        elif score >= 5:
            return "Suggestion: Monitor symptoms. If persistent or worsening, consult healthcare provider."
        else:
            return "Continue self-monitoring. Maintain healthy lifestyle practices."


def analyze_clinical_validity(text: str) -> Dict[str, Any]:
    """
    Complete clinical validity analysis combining DSM-5 and PHQ-9.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Comprehensive clinical assessment
    """
    # Detect DSM-5 symptoms
    dsm5_results = DSM5Criteria.detect_symptoms(text)
    
    # Estimate PHQ-9 score
    phq9_results = PHQ9Estimator.estimate_score(dsm5_results)
    
    return {
        'dsm5_assessment': dsm5_results,
        'phq9_estimation': phq9_results,
        'clinical_summary': {
            'meets_diagnostic_criteria': dsm5_results['meets_dsm5_criteria'],
            'severity': phq9_results['severity_level'],
            'phq9_score': phq9_results['estimated_score'],
            'crisis_risk': dsm5_results['crisis_risk'],
            'recommendation': phq9_results['recommendation']
        }
    }
