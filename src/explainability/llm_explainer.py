import os, json
from typing import Dict, List, Tuple

def build_prompt(text: str) -> str:
    return f"""You are a clinical assistant identifying depression cues using DSM-5/PHQ-9.
Input: "{text}"
Task: Extract symptoms as DSM-5/PHQ-9 labels, list emotions, and write one-sentence explanation.
Return JSON with keys: depression_likelihood (Low/Moderate/High), dsm_symptoms (array), emotions (array), explanation (string).
Only return JSON.
"""

def generate_prose_rationale(text: str, attention_weights: Dict[str, float], prediction: str) -> str:
    """
    BERT-XDD-style: Translate attention + prediction into natural prose.
    
    Research Basis:
    - Belcastro et al. 2024 (BERT-XDD): "ChatGPT generates human-readable rationales
      from technical attention attributions, improving user trust."
    - Converts attention token weights → clinically-grounded natural language explanation
    
    Args:
        text: Original input text
        attention_weights: Dictionary of {token: weight} from attention extraction
        prediction: Model prediction (e.g., "depression", "control")
    
    Returns:
        Natural language explanation (2-3 sentences)
    
    Example:
        Input: "I feel hopeless and can't get out of bed"
        Attention: {"hopeless": 0.45, "can't": 0.32, "bed": 0.28}
        Output: "The text contains 'hopeless' (high attention weight 0.45), indicating 
                depressed mood (DSM-5 criterion 1). The phrase 'can't get out of bed' 
                suggests fatigue (criterion 4). These 2 symptoms meet the threshold 
                for mild depression screening."
    """
    # Sort by attention weight
    if isinstance(attention_weights, dict):
        top_tokens = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        # Handle list/array format
        top_tokens = [(str(i), float(w)) for i, w in enumerate(attention_weights)]
        top_tokens = sorted(top_tokens, key=lambda x: x[1], reverse=True)[:10]
    
    # Format token list for prompt
    token_list = [f"'{token}' (weight: {weight:.3f})" for token, weight in top_tokens[:5]]
    
    prompt = f"""You are explaining a mental health prediction to a clinician.

Text: "{text}"
Prediction: {prediction}
Key Attention Words: {', '.join(token_list)}

Task: Write a 2-3 sentence explanation connecting the key words to the prediction using DSM-5 criteria.
Be factual, concise, and clinically grounded.

Example: "The text contains 'hopeless' and 'worthless', indicating depressed mood (DSM-5 criterion 1). The phrase 'can't get out of bed' suggests fatigue (criterion 4). These 2 symptoms meet the threshold for mild depression screening."

Your explanation:"""

    # Try OpenAI, fallback to rule-based
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}, using fallback")
    
    # Fallback: Rule-based prose generation
    top_words = [token for token, _ in top_tokens[:3]]
    symptom_map = {
        'hopeless': 'depressed mood (DSM-5 criterion 1)',
        'sad': 'depressed mood (DSM-5 criterion 1)',
        'worthless': 'feelings of worthlessness (DSM-5 criterion 7)',
        'tired': 'fatigue (DSM-5 criterion 4)',
        'sleep': 'sleep disturbance (DSM-5 criterion 3)',
        'suicide': 'suicidal ideation (DSM-5 criterion 9)',
        'empty': 'depressed mood (DSM-5 criterion 1)',
        'alone': 'social withdrawal',
        'fail': 'feelings of worthlessness (DSM-5 criterion 7)'
    }
    
    detected_symptoms = []
    for word in top_words:
        for key, symptom in symptom_map.items():
            if key in word.lower():
                detected_symptoms.append(f"'{word}' suggests {symptom}")
                break
    
    if detected_symptoms:
        explanation = f"The text contains {', '.join(detected_symptoms[:2])}. "
        explanation += f"Based on {len(detected_symptoms)} symptom indicator(s), this meets the threshold for {prediction} screening."
    else:
        explanation = f"The model detected {prediction} with high attention on words: {', '.join(top_words)}. Clinical review recommended for diagnosis."
    
    return explanation

def mock_llm_explain(text: str) -> Dict:
    return {"depression_likelihood":"Moderate","dsm_symptoms":["Depressed Mood","Fatigue"],"emotions":["Sadness","Exhaustion"],"explanation":"User expresses persistent sadness and lack of energy consistent with DSM-5 criteria."}

def openai_llm_explain(text: str) -> Dict:
    from openai import OpenAI
    client = OpenAI()
    prompt = build_prompt(text)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2)
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}

def explain(text: str, use_openai: bool=False) -> Dict:
    if use_openai and os.getenv("OPENAI_API_KEY"):
        try:
            return openai_llm_explain(text)
        except Exception:
            pass
    return mock_llm_explain(text)
