"""
Explainability Metrics for Depression Detection

Implements plausibility and faithfulness metrics from hate-speech project:

Plausibility (Do explanations match human rationales?):
- Token-F1: F1 score between predicted and human rationales
- IOU-F1: Intersection-over-Union F1
- AUPRC: Area under precision-recall curve

Faithfulness (Do explanations reflect model behavior?):
- Sufficiency: P(original) - P(only_rationales)
- Comprehensiveness: P(original) - P(without_rationales)

Reference: endsem_pres.pdf (IIT Bombay hate speech project)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from sklearn.metrics import f1_score, precision_recall_curve, auc
import logging

logger = logging.getLogger(__name__)


class ExplainabilityMetrics:
    """
    Compute plausibility and faithfulness metrics for explanations.
    """
    
    @staticmethod
    def token_f1(predicted_rationale: np.ndarray, 
                 human_rationale: np.ndarray) -> float:
        """
        Compute Token-F1: F1 score between predicted and human rationales.
        
        Args:
            predicted_rationale: Binary array (1=important, 0=not)
            human_rationale: Binary array (1=important, 0=not)
        
        Returns:
            F1 score
        """
        # Handle edge cases
        if len(predicted_rationale) != len(human_rationale):
            raise ValueError("Rationale lengths must match")
        
        if predicted_rationale.sum() == 0 and human_rationale.sum() == 0:
            return 1.0  # Both empty = perfect match
        
        if predicted_rationale.sum() == 0 or human_rationale.sum() == 0:
            return 0.0  # One empty, one not = no match
        
        # Compute F1
        f1 = f1_score(human_rationale, predicted_rationale, zero_division=0)
        return float(f1)
    
    @staticmethod
    def iou_f1(predicted_rationale: np.ndarray,
               human_rationale: np.ndarray) -> float:
        """
        Compute IOU-F1: F1 based on intersection-over-union.
        
        IOU = |A ∩ B| / |A ∪ B|
        
        Args:
            predicted_rationale: Binary array
            human_rationale: Binary array
        
        Returns:
            IOU-F1 score
        """
        intersection = np.logical_and(predicted_rationale, human_rationale).sum()
        union = np.logical_or(predicted_rationale, human_rationale).sum()
        
        if union == 0:
            return 1.0  # Both empty
        
        iou = intersection / union
        
        # Convert IOU to F1-like metric
        # F1 = 2 * (precision * recall) / (precision + recall)
        # When IOU is used as both precision and recall:
        # F1 = 2 * IOU / (1 + IOU)
        if iou == 0:
            return 0.0
        
        iou_f1 = 2 * iou / (1 + iou)
        return float(iou_f1)
    
    @staticmethod
    def auprc(predicted_scores: np.ndarray,
              human_rationale: np.ndarray) -> float:
        """
        Compute AUPRC: Area under precision-recall curve.
        
        Args:
            predicted_scores: Continuous importance scores (0-1)
            human_rationale: Binary array (0 or 1)
        
        Returns:
            AUPRC score
        """
        if len(predicted_scores) != len(human_rationale):
            raise ValueError("Lengths must match")
        
        if human_rationale.sum() == 0:
            return 0.0  # No positive samples
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(human_rationale, predicted_scores)
        
        # Compute area under curve
        auprc_score = auc(recall, precision)
        return float(auprc_score)
    
    @staticmethod
    def sufficiency(model_fn: Callable,
                   original_text: str,
                   rationale_tokens: List[str],
                   tokenizer,
                   predicted_class: int) -> float:
        """
        Sufficiency: How much does keeping only rationale tokens preserve prediction?
        
        Sufficiency = P(original) - P(only_rationales)
        
        Lower is better (rationales are sufficient to preserve prediction).
        
        Args:
            model_fn: Function that takes text and returns class probabilities
            original_text: Full text
            rationale_tokens: List of important tokens
            tokenizer: Tokenizer
            predicted_class: The predicted class index
        
        Returns:
            Sufficiency score (0 = perfect, higher = worse)
        """
        # Get original prediction probability
        prob_original = model_fn(original_text)[predicted_class]
        
        # Create text with only rationale tokens
        rationale_text = ' '.join(rationale_tokens)
        
        # Get prediction on rationale-only text
        prob_rationale = model_fn(rationale_text)[predicted_class]
        
        # Sufficiency: how much probability drops when using only rationales
        sufficiency_score = prob_original - prob_rationale
        
        return float(sufficiency_score)
    
    @staticmethod
    def comprehensiveness(model_fn: Callable,
                         original_text: str,
                         rationale_tokens: List[str],
                         tokenizer,
                         predicted_class: int) -> float:
        """
        Comprehensiveness: How much does removing rationale tokens change prediction?
        
        Comprehensiveness = P(original) - P(without_rationales)
        
        Higher is better (removing rationales significantly changes prediction).
        
        Args:
            model_fn: Function that takes text and returns class probabilities
            original_text: Full text
            rationale_tokens: List of important tokens to remove
            tokenizer: Tokenizer
            predicted_class: The predicted class index
        
        Returns:
            Comprehensiveness score (higher = better)
        """
        # Get original prediction probability
        prob_original = model_fn(original_text)[predicted_class]
        
        # Create text without rationale tokens
        tokens = original_text.split()
        rationale_set = set(rationale_tokens)
        non_rationale_tokens = [t for t in tokens if t not in rationale_set]
        non_rationale_text = ' '.join(non_rationale_tokens)
        
        # Handle edge case: all tokens removed
        if not non_rationale_text.strip():
            return prob_original  # Maximum comprehensiveness
        
        # Get prediction without rationales
        prob_without = model_fn(non_rationale_text)[predicted_class]
        
        # Comprehensiveness: how much probability drops when removing rationales
        comprehensiveness_score = prob_original - prob_without
        
        return float(comprehensiveness_score)
    
    @staticmethod
    def compute_all_plausibility(predicted_rationales: List[np.ndarray],
                                 human_rationales: List[np.ndarray],
                                 predicted_scores: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        Compute all plausibility metrics for a dataset.
        
        Args:
            predicted_rationales: List of binary arrays
            human_rationales: List of binary arrays
            predicted_scores: Optional list of continuous scores for AUPRC
        
        Returns:
            Dictionary with Token-F1, IOU-F1, AUPRC (if scores provided)
        """
        token_f1_scores = []
        iou_f1_scores = []
        auprc_scores = []
        
        for i, (pred_rat, human_rat) in enumerate(zip(predicted_rationales, human_rationales)):
            # Token-F1
            tf1 = ExplainabilityMetrics.token_f1(pred_rat, human_rat)
            token_f1_scores.append(tf1)
            
            # IOU-F1
            iou = ExplainabilityMetrics.iou_f1(pred_rat, human_rat)
            iou_f1_scores.append(iou)
            
            # AUPRC (if scores provided)
            if predicted_scores is not None:
                auprc = ExplainabilityMetrics.auprc(predicted_scores[i], human_rat)
                auprc_scores.append(auprc)
        
        results = {
            'token_f1_mean': np.mean(token_f1_scores),
            'token_f1_std': np.std(token_f1_scores),
            'iou_f1_mean': np.mean(iou_f1_scores),
            'iou_f1_std': np.std(iou_f1_scores)
        }
        
        if predicted_scores is not None:
            results['auprc_mean'] = np.mean(auprc_scores)
            results['auprc_std'] = np.std(auprc_scores)
        
        return results
    
    @staticmethod
    def compute_all_faithfulness(model_fn: Callable,
                                 texts: List[str],
                                 rationale_tokens_list: List[List[str]],
                                 tokenizer,
                                 predicted_classes: List[int]) -> Dict[str, float]:
        """
        Compute all faithfulness metrics for a dataset.
        
        Args:
            model_fn: Model prediction function
            texts: List of input texts
            rationale_tokens_list: List of rationale token lists
            tokenizer: Tokenizer
            predicted_classes: List of predicted class indices
        
        Returns:
            Dictionary with Sufficiency and Comprehensiveness means
        """
        sufficiency_scores = []
        comprehensiveness_scores = []
        
        for text, rationale_tokens, pred_class in zip(texts, rationale_tokens_list, predicted_classes):
            # Sufficiency
            suff = ExplainabilityMetrics.sufficiency(
                model_fn, text, rationale_tokens, tokenizer, pred_class
            )
            sufficiency_scores.append(suff)
            
            # Comprehensiveness
            comp = ExplainabilityMetrics.comprehensiveness(
                model_fn, text, rationale_tokens, tokenizer, pred_class
            )
            comprehensiveness_scores.append(comp)
        
        return {
            'sufficiency_mean': np.mean(sufficiency_scores),
            'sufficiency_std': np.std(sufficiency_scores),
            'comprehensiveness_mean': np.mean(comprehensiveness_scores),
            'comprehensiveness_std': np.std(comprehensiveness_scores)
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EXPLAINABILITY METRICS - EXAMPLES")
    print("=" * 80)
    
    # Example 1: Token-F1
    print("\n1. TOKEN-F1")
    print("-" * 80)
    
    predicted = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    human = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    
    token_f1 = ExplainabilityMetrics.token_f1(predicted, human)
    print(f"Predicted rationale: {predicted}")
    print(f"Human rationale:     {human}")
    print(f"Token-F1: {token_f1:.4f}")
    
    # Example 2: IOU-F1
    print("\n2. IOU-F1")
    print("-" * 80)
    
    iou_f1 = ExplainabilityMetrics.iou_f1(predicted, human)
    intersection = np.logical_and(predicted, human).sum()
    union = np.logical_or(predicted, human).sum()
    print(f"Intersection: {intersection}, Union: {union}")
    print(f"IOU-F1: {iou_f1:.4f}")
    
    # Example 3: AUPRC
    print("\n3. AUPRC")
    print("-" * 80)
    
    scores = np.array([0.9, 0.85, 0.6, 0.8, 0.3, 0.2, 0.1, 0.15])
    auprc = ExplainabilityMetrics.auprc(scores, human)
    print(f"Predicted scores: {scores}")
    print(f"Human rationale:  {human}")
    print(f"AUPRC: {auprc:.4f}")
    
    # Example 4: Batch computation
    print("\n4. BATCH PLAUSIBILITY METRICS")
    print("-" * 80)
    
    predicted_batch = [
        np.array([1, 1, 0, 1, 0]),
        np.array([1, 1, 1, 0, 0]),
        np.array([1, 0, 0, 1, 1])
    ]
    
    human_batch = [
        np.array([1, 1, 1, 1, 0]),
        np.array([1, 1, 1, 0, 0]),
        np.array([1, 1, 0, 1, 0])
    ]
    
    scores_batch = [
        np.array([0.9, 0.85, 0.4, 0.8, 0.2]),
        np.array([0.95, 0.9, 0.85, 0.3, 0.25]),
        np.array([0.88, 0.5, 0.3, 0.82, 0.75])
    ]
    
    plausibility = ExplainabilityMetrics.compute_all_plausibility(
        predicted_batch, human_batch, scores_batch
    )
    
    print(f"Token-F1:  {plausibility['token_f1_mean']:.4f} ± {plausibility['token_f1_std']:.4f}")
    print(f"IOU-F1:    {plausibility['iou_f1_mean']:.4f} ± {plausibility['iou_f1_std']:.4f}")
    print(f"AUPRC:     {plausibility['auprc_mean']:.4f} ± {plausibility['auprc_std']:.4f}")
    
    print("\n" + "=" * 80)
    print("USAGE IN EVALUATION:")
    print("=" * 80)
    print("""
    from src.explainability.explainability_metrics import ExplainabilityMetrics
    
    # Compute plausibility metrics
    plausibility = ExplainabilityMetrics.compute_all_plausibility(
        predicted_rationales=model_rationales,
        human_rationales=annotated_rationales,
        predicted_scores=attention_weights
    )
    
    print(f"Token-F1: {plausibility['token_f1_mean']:.4f}")
    print(f"IOU-F1: {plausibility['iou_f1_mean']:.4f}")
    print(f"AUPRC: {plausibility['auprc_mean']:.4f}")
    
    # Compute faithfulness metrics
    def model_predict(text):
        # Your model inference here
        return softmax_probs  # numpy array
    
    faithfulness = ExplainabilityMetrics.compute_all_faithfulness(
        model_fn=model_predict,
        texts=test_texts,
        rationale_tokens_list=rationale_lists,
        tokenizer=tokenizer,
        predicted_classes=predictions
    )
    
    print(f"Sufficiency: {faithfulness['sufficiency_mean']:.4f}")
    print(f"Comprehensiveness: {faithfulness['comprehensiveness_mean']:.4f}")
    """)
