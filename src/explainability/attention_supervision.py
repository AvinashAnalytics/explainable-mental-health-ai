"""
Attention Supervision for Depression Detection

Implements attention loss from hate-speech project:
    L_total = L_class + λ * L_attention
    
where L_attention aligns model attention with human rationales.

Reference: endsem_pres.pdf (IIT Bombay hate speech project)
- Best λ = 100 for explainability
- Best λ = 0.001 for accuracy
- DistilBERT competitive with larger models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionSupervisionLoss(nn.Module):
    """
    Compute attention supervision loss to align model attention with human rationales.
    
    Methods:
    - BCE: Binary cross-entropy between attention and rationale mask
    - MSE: Mean squared error
    - KL: KL divergence (when both are distributions)
    """
    
    def __init__(self, loss_type: str = 'bce', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, 
                attention: torch.Tensor, 
                rationale_mask: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            attention: (batch_size, seq_len) - normalized attention weights
            rationale_mask: (batch_size, seq_len) - binary mask (1=important, 0=not)
            attention_mask: (batch_size, seq_len) - padding mask (1=real, 0=padding)
        
        Returns:
            Loss scalar
        """
        # Apply padding mask if provided
        if attention_mask is not None:
            attention = attention * attention_mask
            rationale_mask = rationale_mask * attention_mask
        
        if self.loss_type == 'bce':
            # Binary cross-entropy (standard approach)
            loss = F.binary_cross_entropy(
                attention, 
                rationale_mask.float(), 
                reduction=self.reduction
            )
        
        elif self.loss_type == 'mse':
            # Mean squared error
            loss = F.mse_loss(
                attention, 
                rationale_mask.float(), 
                reduction=self.reduction
            )
        
        elif self.loss_type == 'kl':
            # KL divergence (if both are distributions)
            # Add small epsilon for stability
            eps = 1e-10
            attention_dist = attention + eps
            rationale_dist = rationale_mask.float() + eps
            
            loss = F.kl_div(
                torch.log(attention_dist),
                rationale_dist,
                reduction=self.reduction
            )
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss


def extract_attention_weights(model, 
                              input_ids: torch.Tensor,
                              attention_mask: torch.Tensor,
                              layer_idx: int = -1,
                              pooling: str = 'mean') -> torch.Tensor:
    """
    Extract attention weights from a transformer model.
    
    Args:
        model: HuggingFace model (DistilBERT, BERT, etc.)
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)
        layer_idx: Which layer to extract (-1 = last layer)
        pooling: How to pool multi-head attention ('mean', 'max', 'first')
    
    Returns:
        attention_weights: (batch_size, seq_len) - normalized per sample
    """
    # Forward pass with output_attentions=True
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    
    # Get attention from specified layer
    # attentions: tuple of (batch_size, num_heads, seq_len, seq_len)
    attentions = outputs.attentions[layer_idx]
    
    # Pool across heads
    if pooling == 'mean':
        # Average across heads
        attention = attentions.mean(dim=1)  # (batch, seq, seq)
    elif pooling == 'max':
        attention = attentions.max(dim=1)[0]
    elif pooling == 'first':
        attention = attentions[:, 0, :, :]  # First head only
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    
    # Get attention to [CLS] token (or sum across query tokens)
    # Option 1: Attention from [CLS] token
    cls_attention = attention[:, 0, :]  # (batch, seq)
    
    # Normalize per sample (softmax already applied, but renormalize after masking)
    cls_attention = cls_attention * attention_mask.float()
    cls_attention = cls_attention / (cls_attention.sum(dim=1, keepdim=True) + 1e-10)
    
    return cls_attention


class AttentionSupervisedTrainer:
    """
    Trainer that incorporates attention supervision loss.
    
    Usage:
        trainer = AttentionSupervisedTrainer(model, tokenizer, lambda_weight=1.0)
        trainer.train(train_dataset, val_dataset, epochs=5)
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 lambda_weight: float = 1.0,
                 attention_loss_type: str = 'bce',
                 attention_layer: int = -1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.lambda_weight = lambda_weight
        self.device = device
        
        self.attention_loss_fn = AttentionSupervisionLoss(loss_type=attention_loss_type)
        self.attention_layer = attention_layer
        
        logger.info(f"Initialized AttentionSupervisedTrainer (λ={lambda_weight})")
    
    def compute_loss(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    labels: torch.Tensor,
                    rationale_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss = classification loss + λ * attention loss.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,)
            rationale_masks: (batch_size, seq_len) - optional human rationales
        
        Returns:
            total_loss, loss_dict
        """
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True
        )
        
        # Classification loss
        loss_class = outputs.loss
        
        # Attention loss (if rationales provided)
        loss_attention = torch.tensor(0.0, device=self.device)
        
        if rationale_masks is not None and self.lambda_weight > 0:
            # Extract attention weights
            attentions = outputs.attentions[self.attention_layer]
            
            # Pool across heads (mean)
            attention = attentions.mean(dim=1)  # (batch, seq, seq)
            
            # Get attention from [CLS] token
            cls_attention = attention[:, 0, :]  # (batch, seq)
            
            # Normalize
            cls_attention = cls_attention * attention_mask.float()
            cls_attention = cls_attention / (cls_attention.sum(dim=1, keepdim=True) + 1e-10)
            
            # Compute attention loss
            loss_attention = self.attention_loss_fn(
                cls_attention,
                rationale_masks,
                attention_mask
            )
        
        # Total loss
        loss_total = loss_class + self.lambda_weight * loss_attention
        
        return loss_total, {
            'loss_total': loss_total.item(),
            'loss_class': loss_class.item(),
            'loss_attention': loss_attention.item()
        }
    
    def train_epoch(self, train_loader, optimizer, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'loss_total': 0.0,
            'loss_class': 0.0,
            'loss_attention': 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            rationale_masks = batch.get('rationale_masks', None)
            
            if rationale_masks is not None:
                rationale_masks = rationale_masks.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                input_ids, attention_mask, labels, rationale_masks
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss_dict['loss_total']:.4f} "
                    f"(class: {loss_dict['loss_class']:.4f}, "
                    f"attn: {loss_dict['loss_attention']:.4f})"
                )
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses


def create_rationale_mask(tokenized_text: Dict,
                          important_tokens: List[str],
                          tokenizer) -> torch.Tensor:
    """
    Create binary rationale mask from list of important tokens.
    
    Args:
        tokenized_text: Output from tokenizer (with 'input_ids')
        important_tokens: List of important words/tokens
        tokenizer: HuggingFace tokenizer
    
    Returns:
        rationale_mask: (seq_len,) binary tensor
    """
    input_ids = tokenized_text['input_ids']
    seq_len = len(input_ids)
    
    rationale_mask = torch.zeros(seq_len)
    
    # Convert important tokens to token IDs
    important_token_ids = set()
    for token in important_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        important_token_ids.update(token_ids)
    
    # Mark positions with important tokens
    for i, token_id in enumerate(input_ids):
        if token_id in important_token_ids:
            rationale_mask[i] = 1.0
    
    return rationale_mask


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("ATTENTION SUPERVISION - EXAMPLE")
    print("=" * 80)
    
    # Mock example
    batch_size = 4
    seq_len = 32
    
    # Simulated attention weights (normalized)
    attention = torch.softmax(torch.randn(batch_size, seq_len), dim=1)
    
    # Simulated human rationales
    rationale_mask = torch.zeros(batch_size, seq_len)
    rationale_mask[:, 5:10] = 1.0  # Tokens 5-10 are important
    
    # Compute loss
    loss_fn = AttentionSupervisionLoss(loss_type='bce')
    loss = loss_fn(attention, rationale_mask)
    
    print(f"\nAttention supervision loss (BCE): {loss.item():.4f}")
    print(f"Attention on rationale tokens: {attention[:, 5:10].mean().item():.4f}")
    print(f"Attention on other tokens: {attention[:, :5].mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("USAGE IN TRAINING:")
    print("=" * 80)
    print("""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Initialize trainer with attention supervision
    trainer = AttentionSupervisedTrainer(
        model=model,
        tokenizer=tokenizer,
        lambda_weight=1.0,  # Try 0.001, 1, 100
        attention_loss_type='bce'
    )
    
    # Train (assumes train_loader provides rationale_masks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(5):
        losses = trainer.train_epoch(train_loader, optimizer, epoch)
        print(f"Epoch {epoch}: {losses}")
    """)
