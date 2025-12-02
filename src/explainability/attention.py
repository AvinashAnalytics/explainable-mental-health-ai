import torch
import numpy as np
from typing import List, Tuple, Any

class AttentionExplainer:
    """
    Extracts attention weights from Transformer models.
    Implements simple attention aggregation (mean over heads).
    """
    
    @staticmethod
    def extract_top_tokens(model: Any, tokenizer: Any, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # Get attentions from last layer: (batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions[-1]
        
        # Average over heads: (batch, seq_len, seq_len)
        avg_attn = attentions.mean(dim=1)
        
        # Focus on [CLS] token attention (index 0) to other tokens
        # Shape: (seq_len)
        cls_attn = avg_attn[0, 0, :]
        
        # Normalize
        cls_attn = cls_attn / cls_attn.sum()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        scores = cls_attn.cpu().numpy()
        
        # Filter special tokens
        token_scores = []
        for token, score in zip(tokens, scores):
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                token_scores.append((token, float(score)))
                
        # Sort by score descending
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        return token_scores[:top_k]
