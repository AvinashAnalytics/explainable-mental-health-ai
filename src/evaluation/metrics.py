from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Union, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation metrics for depression detection models.
    
    Provides:
    - Classification metrics: accuracy, precision, recall, F1, AUC
    - Confusion matrix analysis
    - Explanation quality assessment
    """
    
    @staticmethod
    def compute_classification_metrics(
        y_true: List[int], 
        y_pred: List[int], 
        y_probs: Optional[Union[List[List[float]], List[float]]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_probs: Predicted probabilities (either [[prob_0, prob_1], ...] or [prob_1, ...])
        
        Returns:
            Dictionary with accuracy, precision, recall, F1, AUC, specificity, NPV, PPV
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} != {len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("Cannot compute metrics on empty arrays")
        
        # Check if all values are binary
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        if not unique_true.issubset({0, 1}) or not unique_pred.issubset({0, 1}):
            logger.warning(f"Non-binary values detected. True: {unique_true}, Pred: {unique_pred}")
        
        # Compute basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        
        # Compute confusion matrix for additional metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Positive Predictive Value (same as precision, included for completeness)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "npv": float(npv),
            "ppv": float(ppv),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        
        # Compute AUC if probabilities provided
        if y_probs is not None:
            try:
                # Validate probabilities
                if len(y_probs) != len(y_true):
                    raise ValueError(f"y_probs length {len(y_probs)} != y_true length {len(y_true)}")
                
                # Handle different probability formats
                if isinstance(y_probs[0], (list, tuple, np.ndarray)):
                    # Format: [[prob_0, prob_1], ...]
                    if len(y_probs[0]) != 2:
                        raise ValueError(f"Expected 2 probabilities per sample, got {len(y_probs[0])}")
                    probs_positive = [float(p[1]) for p in y_probs]
                else:
                    # Format: [prob_1, ...]
                    probs_positive = [float(p) for p in y_probs]
                
                # Validate probability values
                if any(p < 0 or p > 1 for p in probs_positive):
                    logger.warning("Some probabilities are outside [0, 1] range")
                
                # Compute AUC
                auc = roc_auc_score(y_true, probs_positive)
                metrics["auc"] = float(auc)
                
            except ValueError as e:
                logger.error(f"Failed to compute AUC: {e}")
                metrics["auc"] = None
            except Exception as e:
                logger.error(f"Unexpected error computing AUC: {e}")
                metrics["auc"] = None
        
        return metrics

    @staticmethod
    def evaluate_explanation_fluency(explanation: str) -> float:
        """
        Evaluate the fluency/quality of an explanation.
        
        Improved heuristic based on:
        - Length (optimal: 10-50 words)
        - Sentence structure
        - Keyword presence
        
        Args:
            explanation: Text explanation to evaluate
        
        Returns:
            Fluency score between 0.0 and 1.0
        """
        if not explanation or not explanation.strip():
            return 0.0
        
        explanation = explanation.strip()
        words = explanation.split()
        word_count = len(words)
        
        # Length score (optimal: 10-50 words)
        if word_count < 5:
            length_score = word_count / 5.0  # Penalize very short
        elif word_count <= 50:
            length_score = 1.0  # Optimal range
        else:
            length_score = max(0.5, 1.0 - (word_count - 50) / 100.0)  # Penalize very long
        
        # Sentence structure score (has punctuation and capitalization)
        structure_score = 0.0
        if explanation[0].isupper():  # Starts with capital
            structure_score += 0.3
        if any(p in explanation for p in '.!?'):  # Has ending punctuation
            structure_score += 0.3
        if ',' in explanation:  # Has internal punctuation (complex sentences)
            structure_score += 0.2
        if explanation.count('.') <= 5:  # Not too many sentences
            structure_score += 0.2
        
        # Clinical keyword score (relevant for mental health)
        clinical_keywords = [
            'symptom', 'depression', 'mood', 'anhedonia', 'fatigue', 'sleep',
            'dsm', 'phq', 'severity', 'criteria', 'risk', 'suicid', 'hopeless',
            'anxiety', 'stress', 'mental health', 'clinical', 'diagnos'
        ]
        explanation_lower = explanation.lower()
        keyword_count = sum(1 for kw in clinical_keywords if kw in explanation_lower)
        keyword_score = min(1.0, keyword_count / 3.0)  # Optimal: 3+ keywords
        
        # Combined score (weighted average)
        fluency_score = (
            length_score * 0.4 +
            structure_score * 0.3 +
            keyword_score * 0.3
        )
        
        return min(1.0, max(0.0, fluency_score))
    
    @staticmethod
    def get_confusion_matrix_dict(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        """
        Get confusion matrix as dictionary.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with tn, fp, fn, tp counts
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
