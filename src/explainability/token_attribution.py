"""
Token Attribution for Depression Detection Models

This module implements Integrated Gradients (IG) for faithful token-level explanations
in transformer-based text classifiers.

Research Basis:
- Sundararajan et al. 2017: "Axiomatic Attribution for Deep Networks"
- IG provides theoretically grounded attributions with completeness & sensitivity axioms
- Preferred over attention weights which are NOT faithful explanations
  (Jain & Wallace 2019, Serrano & Smith 2019, Wiegreffe & Pinter 2019)

Why Integrated Gradients?
1. Completeness: Attributions sum to (prediction - baseline_prediction)
2. Sensitivity: If input differs, attribution is non-zero
3. Implementation Invariance: Functionally equivalent networks get same attributions
4. Widely adopted in XAI research (Google Captum, mental health NLP)

Mental Health Context:
- Used in arXiv:2304.03347 for mental health LLM interpretability
- Provides faithful explanations crucial for clinical trust
- Avoids misleading attention-based "explanations"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TokenAttributionExplainer:
    """
    Compute faithful token attributions using Integrated Gradients.
    
    This class provides a complete pipeline:
    1. Compute IG attributions for each token
    2. Merge subword tokens back to words
    3. Normalize scores within sentence
    4. Bucket into high/medium/low importance levels
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = 'cpu',
        n_steps: int = 50
    ):
        """
        Initialize token attribution explainer.
        
        Args:
            model: PyTorch transformer model
            tokenizer: Tokenizer (BERT, RoBERTa, DistilBERT)
            device: 'cpu' or 'cuda'
            n_steps: Number of integration steps (default 50, paper uses 20-300)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Thresholds for bucketing normalized scores
        self.HIGH_THRESHOLD = 0.75    # Top 25% → high (red)
        self.MEDIUM_THRESHOLD = 0.40  # Middle 35% → medium (yellow)
        # Below 0.40 → low (green)
        
    def explain_text(
        self,
        text: str,
        prediction_label: int,
        return_all_tokens: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate complete token-level explanation for input text.
        
        Args:
            text: Input text to explain
            prediction_label: Predicted class (0=control, 1=depression)
            return_all_tokens: If True, return all tokens; if False, return top 10
        
        Returns:
            List of dicts with structure:
            [
                {"word": "restless", "score": 0.92, "level": "high"},
                {"word": "pacing", "score": 0.71, "level": "medium"},
                ...
            ]
        """
        try:
            # Step 1: Compute raw IG attributions
            attributions, tokens = self._compute_attributions(text, prediction_label)
            
            # Step 2: Merge subwords into words
            word_scores = self._merge_subwords(tokens, attributions)
            
            # Step 3: Normalize scores
            normalized_scores = self._normalize_scores(word_scores)
            
            # Step 4: Bucket into importance levels
            bucketed_tokens = self._bucket_scores(normalized_scores)
            
            # Step 5: Filter and sort
            # Remove very short words and punctuation
            bucketed_tokens = [
                t for t in bucketed_tokens 
                if len(t['word']) > 1 and t['word'].isalnum()
            ]
            
            # Sort by score descending
            bucketed_tokens = sorted(bucketed_tokens, key=lambda x: x['score'], reverse=True)
            
            # Return top 10 unless all requested
            if not return_all_tokens:
                bucketed_tokens = bucketed_tokens[:10]
            
            return bucketed_tokens
            
        except NotImplementedError as e:
            logger.error(f"Model not supported: {e}")
            raise  # Re-raise to trigger fallback
        except Exception as e:
            logger.error(f"Error in explain_text: {e}", exc_info=True)
            # Print more details for debugging
            import traceback
            print("=" * 80)
            print("DETAILED ERROR IN INTEGRATED GRADIENTS:")
            print(traceback.format_exc())
            print("=" * 80)
            print(f"Text length: {len(text)} chars")
            print(f"Prediction label: {prediction_label}")
            print(f"Model type: {type(self.model).__name__}")
            print("=" * 80)
            return []  # Return empty instead of crashing
    
    def _compute_attributions(
        self,
        text: str,
        target_class: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            text: Input text
            target_class: Class to compute attributions for (0 or 1)
        
        Returns:
            (attributions, tokens) where attributions is array of shape (seq_len,)
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create baseline (all PAD tokens)
        baseline_inputs = {
            'input_ids': torch.full_like(inputs['input_ids'], self.tokenizer.pad_token_id),
            'attention_mask': torch.zeros_like(inputs['attention_mask'])
        }
        
        # Get embeddings
        input_embeddings = self._get_embeddings(inputs)
        baseline_embeddings = self._get_embeddings(baseline_inputs)
        
        # Compute integrated gradients
        attributions = self._integrated_gradients(
            input_embeddings,
            baseline_embeddings,
            inputs,
            target_class
        )
        
        # Sum over embedding dimension → (batch, seq_len)
        token_attributions = attributions.sum(dim=-1).squeeze(0).cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return token_attributions, tokens
    
    def _get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embeddings from input_ids."""
        if hasattr(self.model, 'bert'):
            embeddings = self.model.bert.embeddings.word_embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'roberta'):
            embeddings = self.model.roberta.embeddings.word_embeddings(inputs['input_ids'])
        elif hasattr(self.model, 'distilbert'):
            embeddings = self.model.distilbert.embeddings.word_embeddings(inputs['input_ids'])
        else:
            # Generic fallback
            embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
        return embeddings
    
    def _integrated_gradients(
        self,
        input_embeddings: torch.Tensor,
        baseline_embeddings: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        target_class: int
    ) -> torch.Tensor:
        """
        Compute integrated gradients along path from baseline to input.
        
        Formula: IG = (x - x') * ∫[α=0 to 1] ∇F(x' + α(x - x')) dα
        
        Approximated as: IG ≈ (x - x') * (1/m) * Σ[i=1 to m] ∇F(x' + (i/m)(x - x'))
        """
        # Compute interpolated inputs
        # alphas shape: (n_steps,)
        alphas = torch.linspace(0, 1, self.n_steps + 1).to(self.device)
        
        # Scale difference
        delta = input_embeddings - baseline_embeddings
        
        # Accumulate gradients
        gradient_sum = torch.zeros_like(input_embeddings)
        
        for alpha in alphas[1:]:  # Skip alpha=0
            # Interpolated embedding: x' + α(x - x')
            interpolated = baseline_embeddings + alpha * delta
            # Clone and detach to create a leaf variable
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            outputs = self._forward_with_embeddings(interpolated, inputs)
            
            # Get logit for target class
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            target_logit = logits[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            target_logit.backward(retain_graph=False)
            
            # Accumulate gradients
            gradient_sum += interpolated.grad
        
        # Average gradients and multiply by delta
        avg_gradients = gradient_sum / self.n_steps
        attributions = delta * avg_gradients
        
        return attributions.detach()
    
    def _forward_with_embeddings(
        self,
        embeddings: torch.Tensor,
        inputs: Dict[str, torch.Tensor]
    ) -> Any:
        """
        Forward pass using provided embeddings instead of input_ids.
        """
        attention_mask = inputs['attention_mask']
        
        # Convert attention mask to proper format for transformer models
        # BERT expects: [batch_size, 1, 1, seq_length] with dtype float
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=embeddings.dtype)  # Match embedding dtype
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(embeddings.dtype).min
        
        try:
            if hasattr(self.model, 'bert'):
                encoder_outputs = self.model.bert.encoder(
                    embeddings,
                    attention_mask=extended_attention_mask
                )
                pooled_output = self.model.bert.pooler(encoder_outputs[0])
                outputs = self.model.classifier(pooled_output)
                
            elif hasattr(self.model, 'roberta'):
                encoder_outputs = self.model.roberta.encoder(
                    embeddings,
                    attention_mask=extended_attention_mask
                )
                sequence_output = encoder_outputs[0]  # Keep 3D: (batch, seq_len, hidden)
                outputs = self.model.classifier(sequence_output)  # Classifier handles pooling internally
                
            elif hasattr(self.model, 'distilbert'):
                # DistilBERT: embeddings → transformer → CLS → pre_classifier → ReLU → dropout → classifier
                # Create head_mask (all None for all layers)
                num_layers = len(self.model.distilbert.transformer.layer)
                head_mask = [None] * num_layers
                
                # Convert attention mask to float (DistilBERT requires float or bool, not int64)
                attn_mask_float = attention_mask.float()
                
                encoder_outputs = self.model.distilbert.transformer(
                    embeddings,
                    attn_mask=attn_mask_float,  # DistilBERT uses 'attn_mask' not 'attention_mask'
                    head_mask=head_mask
                )
                
                # Get CLS token
                cls_token = encoder_outputs[0][:, 0, :]
                
                # Pass through pre_classifier + ReLU + dropout + classifier
                hidden_state = self.model.pre_classifier(cls_token)
                hidden_state = torch.relu(hidden_state)
                hidden_state = self.model.dropout(hidden_state)
                outputs = self.model.classifier(hidden_state)
            else:
                # Try generic approach with inputs_embeds
                try:
                    outputs = self.model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask
                    )
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    
                    class OutputContainer:
                        def __init__(self, logits):
                            self.logits = logits
                    
                    return OutputContainer(outputs)
                except:
                    raise NotImplementedError(
                        f"Model type {type(self.model).__name__} not fully supported. "
                        f"Supported: BERT, RoBERTa, DistilBERT models. "
                        f"Try using the fallback Gradient×Input method."
                    )
            
            # Return in same format as normal forward
            class OutputContainer:
                def __init__(self, logits):
                    self.logits = logits
            
            return OutputContainer(outputs)
            
        except Exception as e:
            logger.error(f"Error in _forward_with_embeddings: {e}")
            raise
    
    def _merge_subwords(
        self,
        tokens: List[str],
        attributions: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Merge subword tokens back into complete words.
        
        Handles different tokenizer formats:
        - BERT: uses ## for continuations
        - RoBERTa: uses Ġ for word starts
        - SentencePiece: uses ▁ for word starts
        
        Returns:
            List of (word, cumulative_attribution) tuples
        """
        word_scores = []
        current_word = ""
        current_score = 0.0
        
        for token, score in zip(tokens, attributions):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '<unk>', '[UNK]']:
                continue
            
            # BERT: ## indicates continuation
            if token.startswith('##'):
                current_word += token[2:]
                current_score += float(score)
            
            # RoBERTa/SentencePiece: Ġ or ▁ indicates new word
            elif token.startswith('Ġ') or token.startswith('▁'):
                # Save previous word
                if current_word:
                    word_scores.append((current_word, current_score))
                # Start new word
                current_word = token[1:]
                current_score = float(score)
            
            # Start of new word or single character
            else:
                # Save previous word if it exists
                if current_word:
                    word_scores.append((current_word, current_score))
                # Start new word
                current_word = token
                current_score = float(score)
        
        # Add last word
        if current_word:
            word_scores.append((current_word, current_score))
        
        return word_scores
    
    def _normalize_scores(
        self,
        word_scores: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        This ensures scores are comparable across different sentences
        and creates separation between important and unimportant tokens.
        
        Optional: Apply power scaling (score^1.5) to increase separation.
        """
        if not word_scores:
            return []
        
        # Extract scores
        scores = np.array([score for _, score in word_scores])
        
        # Use absolute values (attributions can be negative)
        scores = np.abs(scores)
        
        # Min-max normalization
        min_score = scores.min()
        max_score = scores.max()
        epsilon = 1e-10  # Avoid division by zero
        
        if max_score - min_score < epsilon:
            # All scores are the same - assign 0.5 (medium)
            normalized = np.full_like(scores, 0.5)
        else:
            normalized = (scores - min_score) / (max_score - min_score + epsilon)
        
        # Optional: Apply power scaling to increase separation
        # This makes high scores higher and low scores lower
        normalized = np.power(normalized, 1.5)
        
        # Create normalized word-score pairs
        normalized_pairs = [
            (word, float(norm_score))
            for (word, _), norm_score in zip(word_scores, normalized)
        ]
        
        return normalized_pairs
    
    def _bucket_scores(
        self,
        normalized_scores: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Bucket normalized scores into importance levels.
        
        Thresholds:
        - High (red): score >= 0.75
        - Medium (yellow): 0.40 <= score < 0.75
        - Low (green): score < 0.40
        
        Returns:
            List of dicts: [{"word": str, "score": float, "level": str}, ...]
        """
        bucketed = []
        
        for word, score in normalized_scores:
            if score >= self.HIGH_THRESHOLD:
                level = "High"
            elif score >= self.MEDIUM_THRESHOLD:
                level = "Medium"
            else:
                level = "Low"
            
            bucketed.append({
                "word": word,
                "score": score,
                "level": level
            })
        
        return bucketed


def explain_tokens_with_ig(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    prediction: int,
    device: str = 'cpu',
    n_steps: int = 50
) -> List[Dict[str, Any]]:
    """
    Convenience function to explain tokens using Integrated Gradients.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        text: Input text
        prediction: Predicted class (0 or 1)
        device: 'cpu' or 'cuda'
        n_steps: Number of integration steps
    
    Returns:
        List of token explanations:
        [
            {"word": "restless", "score": 0.92, "level": "high"},
            {"word": "pacing", "score": 0.71, "level": "medium"},
            ...
        ]
    """
    explainer = TokenAttributionExplainer(model, tokenizer, device, n_steps)
    return explainer.explain_text(text, prediction)
