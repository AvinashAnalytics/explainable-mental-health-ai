"""
Safety filters and ethical constraints for mental health AI.

Implements crisis detection, medical claim filtering, and resource routing.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# Crisis indicators (high-risk keywords and patterns)
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 
    'better off dead', 'going to kill', 'plan to die',
    'suicide plan', 'collected pills', 'wrote goodbye',
    'final goodbye', 'can\'t go on', 'no reason to live',
    'self-harm', 'cut myself', 'hurt myself'
]

# Medical claim patterns to filter
MEDICAL_CLAIM_PATTERNS = [
    r'you (have|are diagnosed with|suffer from) .*depression',
    r'this (is|means|indicates) (major )?depressive disorder',
    r'you (need|should take|require) (medication|antidepressants|therapy)',
    r'i (diagnose|prescribe|recommend treatment)',
]

# Disclaimer text
STANDARD_DISCLAIMER = (
    "This analysis is for informational purposes only and does not constitute "
    "medical advice, diagnosis, or treatment. If you are experiencing mental health "
    "concerns, please consult a qualified mental health professional."
)

# Crisis resources
CRISIS_RESOURCES = [
    "🆘 National Suicide Prevention Lifeline: 988 (US)",
    "📱 Crisis Text Line: Text HOME to 741741 (US)",
    "🌍 International Crisis Resources: https://www.iasp.info/resources/Crisis_Centres/",
    "🚨 If this is an emergency, call 911 (US) or your local emergency number"
]


def detect_crisis_risk(text: str) -> Dict[str, any]:
    """
    Detect if text indicates crisis-level mental health risk.
    
    Returns:
        Dict with crisis flag, matched keywords, and risk level
    """
    text_lower = text.lower()
    matched_keywords = []
    
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            matched_keywords.append(keyword)
    
    # Check for negations that reduce risk
    negation_patterns = [
        'not suicidal', 'don\'t want to die', 'would never hurt',
        'not going to', 'won\'t hurt myself'
    ]
    
    has_negation = any(neg in text_lower for neg in negation_patterns)
    
    # Determine risk level
    if matched_keywords and not has_negation:
        if len(matched_keywords) >= 3 or any(k in ['suicide plan', 'going to kill', 'collected pills'] for k in matched_keywords):
            risk_level = 'high'
        elif len(matched_keywords) >= 2:
            risk_level = 'elevated'
        else:
            risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    return {
        'is_crisis': risk_level in ['high', 'elevated'],
        'risk_level': risk_level,
        'matched_keywords': matched_keywords,
        'has_protective_factors': has_negation
    }


def filter_medical_claims(text: str) -> str:
    """
    Remove or flag medical diagnostic statements.
    
    Returns:
        Filtered text with medical claims removed/softened
    """
    filtered = text
    
    for pattern in MEDICAL_CLAIM_PATTERNS:
        filtered = re.sub(pattern, '[MEDICAL CLAIM REMOVED]', filtered, flags=re.IGNORECASE)
    
    return filtered


def apply_safety_layer(analysis_result: Dict) -> Dict:
    """
    Apply safety checks and modifications to analysis results.
    
    Args:
        analysis_result: Raw analysis from model or LLM
        
    Returns:
        Safety-enhanced result with crisis routing and disclaimers
    """
    # Check for crisis indicators
    if 'input_text' in analysis_result:
        crisis_check = detect_crisis_risk(analysis_result['input_text'])
    else:
        crisis_check = {'is_crisis': False, 'risk_level': 'unknown'}
    
    # Override crisis flag if high risk detected
    if crisis_check['is_crisis']:
        analysis_result['requires_crisis_intervention'] = True
        analysis_result['crisis_risk_level'] = crisis_check['risk_level']
        analysis_result['crisis_keywords'] = crisis_check['matched_keywords']
    
    # Add crisis resources if needed
    if analysis_result.get('requires_crisis_intervention', False):
        analysis_result['crisis_resources'] = CRISIS_RESOURCES
        analysis_result['urgent_note'] = (
            "⚠️ CRISIS INDICATORS DETECTED. This individual may be at risk of self-harm. "
            "Please contact crisis services immediately."
        )
    
    # Filter medical claims from explanations
    if 'explanation' in analysis_result:
        analysis_result['explanation'] = filter_medical_claims(analysis_result['explanation'])
    
    if 'clinical_note' in analysis_result:
        analysis_result['clinical_note'] = filter_medical_claims(analysis_result['clinical_note'])
    
    # Add disclaimer
    analysis_result['disclaimer'] = STANDARD_DISCLAIMER
    
    # Add safety metadata
    analysis_result['safety_checks_applied'] = True
    analysis_result['crisis_screening_performed'] = True
    
    return analysis_result


def validate_llm_output(llm_response: Dict) -> Dict:
    """
    Validate LLM output for safety and quality.
    
    Checks:
    - No harmful content generation
    - No medical prescriptions
    - Crisis indicators properly flagged
    - Response format is valid
    """
    validation_result = {
        'is_valid': True,
        'issues': [],
        'severity': 'none'
    }
    
    # Check for harmful content
    response_text = str(llm_response.get('explanation', '')) + str(llm_response.get('clinical_note', ''))
    
    harmful_patterns = [
        'how to commit suicide',
        'methods of self-harm',
        'you should kill',
        'best way to die'
    ]
    
    for pattern in harmful_patterns:
        if pattern in response_text.lower():
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Harmful content detected: {pattern}")
            validation_result['severity'] = 'critical'
    
    # Check for inappropriate medical advice
    for pattern in MEDICAL_CLAIM_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            validation_result['is_valid'] = False
            validation_result['issues'].append("Medical claim/diagnosis detected")
            validation_result['severity'] = 'high' if validation_result['severity'] != 'critical' else 'critical'
    
    # Check crisis flag consistency
    if 'requires_crisis_intervention' in llm_response:
        crisis_flag = llm_response['requires_crisis_intervention']
        actual_crisis = detect_crisis_risk(llm_response.get('input_text', ''))
        
        if actual_crisis['is_crisis'] and not crisis_flag:
            validation_result['issues'].append("Crisis indicators present but not flagged")
            validation_result['severity'] = 'high' if validation_result['severity'] == 'none' else validation_result['severity']
    
    return validation_result


class SafetyGuard:
    """Centralized safety management for mental health AI."""
    
    def __init__(self, enable_crisis_routing: bool = True, enable_filtering: bool = True):
        self.enable_crisis_routing = enable_crisis_routing
        self.enable_filtering = enable_filtering
        logger.info("SafetyGuard initialized")
    
    def process(self, analysis_result: Dict) -> Dict:
        """Apply all safety checks and modifications."""
        # Apply safety layer
        safe_result = apply_safety_layer(analysis_result)
        
        # Validate if from LLM
        if 'model' in analysis_result:
            validation = validate_llm_output(safe_result)
            safe_result['safety_validation'] = validation
            
            if not validation['is_valid']:
                logger.warning(f"Safety validation failed: {validation['issues']}")
                safe_result['safety_warning'] = "Output failed safety checks and has been modified"
        
        return safe_result
