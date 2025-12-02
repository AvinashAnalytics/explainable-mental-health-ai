"""
Developer Tools Module
======================
Advanced debugging and inspection tools for XAI researchers and developers.

Features:
- Raw logits display
- Attention matrices visualization
- Integrated Gradients heatmaps
- Layer-wise activations
- Model internals inspector

Author: Major Project - Explainable AI System
Created: 2025
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DeveloperTools:
    """
    Advanced tools for debugging and inspecting model behavior.
    
    Provides research-grade analysis capabilities including:
    - Raw model outputs (logits, probabilities, hidden states)
    - Attention pattern visualization
    - Layer-wise activation analysis
    - Gradient flow inspection
    - Model architecture exploration
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        """
        Initialize developer tools.
        
        Args:
            model: Transformer model (BERT, RoBERTa, DistilBERT)
            tokenizer: Corresponding tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def extract_raw_logits(self, text: str) -> Dict[str, Any]:
        """
        Extract raw logits and probabilities from model output.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
            - logits: Raw unnormalized scores [batch_size, num_classes]
            - probabilities: Softmax probabilities [batch_size, num_classes]
            - predicted_class: Predicted class index
            - confidence: Confidence score (max probability)
            - class_scores: Per-class probability breakdown
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Compute probabilities
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, predicted_class].item()
            
            # Get all class scores
            class_labels = ['Not Depressed', 'Depressed']  # Binary classification
            class_scores = {
                label: probs[0, idx].item()
                for idx, label in enumerate(class_labels)
            }
            
            return {
                'logits': logits.cpu().numpy().tolist(),
                'probabilities': probs.cpu().numpy().tolist(),
                'predicted_class': predicted_class,
                'predicted_label': class_labels[predicted_class],
                'confidence': confidence,
                'class_scores': class_scores,
                'logit_difference': abs(logits[0, 0].item() - logits[0, 1].item())
            }
            
        except Exception as e:
            logger.error(f"Error extracting logits: {e}")
            return {
                'error': str(e),
                'logits': None,
                'probabilities': None
            }
    
    def extract_attention_matrices(self, text: str) -> Dict[str, Any]:
        """
        Extract attention weights from all layers.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
            - attention_matrices: List of attention tensors per layer
            - num_layers: Number of attention layers
            - num_heads: Number of attention heads per layer
            - tokens: Tokenized input
            - average_attention: Average attention across all heads/layers
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get tokens for display
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Forward pass with attention output
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True
                )
                
            attentions = outputs.attentions  # Tuple of tensors
            
            if attentions is None:
                return {
                    'error': 'Model does not support attention output',
                    'attention_matrices': None
                }
            
            # Process attention matrices
            attention_list = []
            for layer_idx, attn in enumerate(attentions):
                # attn shape: [batch_size, num_heads, seq_len, seq_len]
                attn_np = attn.cpu().numpy()[0]  # Remove batch dimension
                attention_list.append(attn_np)
            
            # Average attention across all heads and layers
            avg_attention = np.mean([np.mean(attn, axis=0) for attn in attention_list], axis=0)
            
            return {
                'attention_matrices': [attn.tolist() for attn in attention_list],
                'num_layers': len(attention_list),
                'num_heads': attention_list[0].shape[0] if attention_list else 0,
                'tokens': tokens,
                'sequence_length': len(tokens),
                'average_attention': avg_attention.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error extracting attention: {e}")
            return {
                'error': str(e),
                'attention_matrices': None
            }
    
    def extract_hidden_states(self, text: str) -> Dict[str, Any]:
        """
        Extract hidden states from all layers.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
            - hidden_states: List of hidden state tensors per layer
            - num_layers: Number of transformer layers
            - hidden_size: Dimension of hidden states
            - tokens: Tokenized input
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Forward pass with hidden states output
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
            hidden_states = outputs.hidden_states  # Tuple of tensors
            
            if hidden_states is None:
                return {
                    'error': 'Model does not support hidden states output',
                    'hidden_states': None
                }
            
            # Process hidden states
            hidden_list = []
            for layer_idx, hidden in enumerate(hidden_states):
                # hidden shape: [batch_size, seq_len, hidden_size]
                hidden_np = hidden.cpu().numpy()[0]  # Remove batch dimension
                
                # Compute statistics for this layer
                layer_stats = {
                    'layer': layer_idx,
                    'mean': float(np.mean(hidden_np)),
                    'std': float(np.std(hidden_np)),
                    'min': float(np.min(hidden_np)),
                    'max': float(np.max(hidden_np)),
                    'norm': float(np.linalg.norm(hidden_np))
                }
                hidden_list.append(layer_stats)
            
            return {
                'hidden_states_stats': hidden_list,
                'num_layers': len(hidden_states),
                'hidden_size': hidden_states[0].shape[-1],
                'sequence_length': len(tokens),
                'tokens': tokens
            }
            
        except Exception as e:
            logger.error(f"Error extracting hidden states: {e}")
            return {
                'error': str(e),
                'hidden_states': None
            }
    
    def analyze_gradient_flow(self, text: str, target_class: int = 1) -> Dict[str, Any]:
        """
        Analyze gradient flow through the model.
        
        Args:
            text: Input text to analyze
            target_class: Class to compute gradients for (0=not depressed, 1=depressed)
            
        Returns:
            Dictionary containing:
            - gradients: Gradient magnitudes per layer
            - gradient_norms: L2 norm of gradients per layer
            - vanishing_gradient_detected: Boolean flag
            - exploding_gradient_detected: Boolean flag
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Enable gradients
            inputs = {k: v.requires_grad_(False) for k, v in inputs.items()}
            
            # Forward pass with hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
            
            # Compute loss for target class
            logits = outputs.logits
            target = torch.tensor([target_class]).to(self.device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Collect gradients from all parameters
            gradient_norms = []
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms.append({
                        'parameter': name,
                        'gradient_norm': grad_norm
                    })
            
            # Sort by gradient magnitude
            gradient_norms = sorted(gradient_norms, key=lambda x: x['gradient_norm'], reverse=True)
            
            # Detect vanishing/exploding gradients
            grad_values = [g['gradient_norm'] for g in gradient_norms]
            vanishing = any(g < 1e-7 for g in grad_values)
            exploding = any(g > 1e3 for g in grad_values)
            
            return {
                'gradient_norms': gradient_norms[:20],  # Top 20
                'max_gradient': max(grad_values) if grad_values else 0,
                'min_gradient': min(grad_values) if grad_values else 0,
                'mean_gradient': np.mean(grad_values) if grad_values else 0,
                'vanishing_gradient_detected': vanishing,
                'exploding_gradient_detected': exploding,
                'total_parameters': len(gradient_norms)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gradient flow: {e}")
            return {
                'error': str(e),
                'gradient_norms': None
            }
    
    def get_model_architecture_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model architecture information.
        
        Returns:
            Dictionary containing:
            - model_type: Type of model (BERT, RoBERTa, etc.)
            - num_parameters: Total trainable parameters
            - num_layers: Number of transformer layers
            - hidden_size: Hidden layer dimension
            - num_attention_heads: Number of attention heads
            - vocab_size: Vocabulary size
            - max_position_embeddings: Maximum sequence length
        """
        try:
            config = self.model.config
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                'model_type': config.model_type if hasattr(config, 'model_type') else 'unknown',
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'num_layers': config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else None,
                'hidden_size': config.hidden_size if hasattr(config, 'hidden_size') else None,
                'num_attention_heads': config.num_attention_heads if hasattr(config, 'num_attention_heads') else None,
                'vocab_size': config.vocab_size if hasattr(config, 'vocab_size') else None,
                'max_position_embeddings': config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else None,
                'intermediate_size': config.intermediate_size if hasattr(config, 'intermediate_size') else None,
                'hidden_dropout_prob': config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else None,
                'attention_probs_dropout_prob': config.attention_probs_dropout_prob if hasattr(config, 'attention_probs_dropout_prob') else None,
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'error': str(e)
            }
    
    def generate_diagnostic_report(self, text: str) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic report for a given text.
        
        Combines all developer tools into a single report.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all diagnostic information
        """
        report = {
            'input_text': text,
            'text_length': len(text),
            'model_info': self.get_model_architecture_info(),
            'logits_analysis': self.extract_raw_logits(text),
            'attention_analysis': self.extract_attention_matrices(text),
            'hidden_states_analysis': self.extract_hidden_states(text),
            'gradient_analysis': self.analyze_gradient_flow(text)
        }
        
        return report


# ============================================================================
# Helper Functions for UI Integration
# ============================================================================

def format_logits_display(logits_data: Dict[str, Any]) -> str:
    """
    Format logits data for display in Streamlit.
    
    Args:
        logits_data: Output from extract_raw_logits()
        
    Returns:
        Formatted HTML string
    """
    if 'error' in logits_data:
        return f"<div style='color: red;'>Error: {logits_data['error']}</div>"
    
    html = "<div style='background: #f0f2f6; padding: 15px; border-radius: 10px;'>"
    html += "<h4>🔢 Raw Model Output</h4>"
    
    # Logits
    html += "<p><strong>Raw Logits:</strong></p>"
    html += "<ul>"
    for idx, logit in enumerate(logits_data['logits'][0]):
        html += f"<li>Class {idx}: {logit:.4f}</li>"
    html += "</ul>"
    
    # Probabilities
    html += "<p><strong>Softmax Probabilities:</strong></p>"
    for label, prob in logits_data['class_scores'].items():
        bar_width = int(prob * 200)
        color = '#ff4d4d' if 'Depressed' in label and prob > 0.5 else '#44cc44'
        html += f"<div style='margin: 5px 0;'>"
        html += f"<strong>{label}:</strong> {prob:.4f}<br>"
        html += f"<div style='background: {color}; width: {bar_width}px; height: 20px; border-radius: 5px;'></div>"
        html += "</div>"
    
    # Confidence metrics
    html += f"<p><strong>Prediction:</strong> {logits_data['predicted_label']}</p>"
    html += f"<p><strong>Confidence:</strong> {logits_data['confidence']:.4f}</p>"
    html += f"<p><strong>Logit Difference:</strong> {logits_data['logit_difference']:.4f}</p>"
    html += "<p style='font-size: 0.9em; color: #666;'><em>Higher logit difference indicates more confident prediction</em></p>"
    
    html += "</div>"
    return html


def format_attention_summary(attention_data: Dict[str, Any]) -> str:
    """
    Format attention analysis summary for display.
    
    Args:
        attention_data: Output from extract_attention_matrices()
        
    Returns:
        Formatted HTML string
    """
    if 'error' in attention_data:
        return f"<div style='color: red;'>Error: {attention_data['error']}</div>"
    
    html = "<div style='background: #f0f2f6; padding: 15px; border-radius: 10px;'>"
    html += "<h4>👁️ Attention Analysis</h4>"
    html += f"<p><strong>Number of Layers:</strong> {attention_data['num_layers']}</p>"
    html += f"<p><strong>Attention Heads per Layer:</strong> {attention_data['num_heads']}</p>"
    html += f"<p><strong>Sequence Length:</strong> {attention_data['sequence_length']} tokens</p>"
    html += "<p><strong>Tokens:</strong></p>"
    html += "<div style='background: white; padding: 10px; border-radius: 5px; font-family: monospace;'>"
    html += " ".join(attention_data['tokens'][:50])  # Show first 50 tokens
    if attention_data['sequence_length'] > 50:
        html += f" ... ({attention_data['sequence_length'] - 50} more)"
    html += "</div>"
    html += "<p style='font-size: 0.9em; color: #666; margin-top: 10px;'>"
    html += "<em>Note: Full attention matrices available in diagnostic report. "
    html += "Use attention heatmap visualization below.</em></p>"
    html += "</div>"
    return html


def format_hidden_states_summary(hidden_data: Dict[str, Any]) -> str:
    """
    Format hidden states analysis for display.
    
    Args:
        hidden_data: Output from extract_hidden_states()
        
    Returns:
        Formatted HTML string
    """
    if 'error' in hidden_data:
        return f"<div style='color: red;'>Error: {hidden_data['error']}</div>"
    
    html = "<div style='background: #f0f2f6; padding: 15px; border-radius: 10px;'>"
    html += "<h4>🧠 Hidden States Analysis</h4>"
    html += f"<p><strong>Number of Layers:</strong> {hidden_data['num_layers']}</p>"
    html += f"<p><strong>Hidden Size:</strong> {hidden_data['hidden_size']}</p>"
    html += f"<p><strong>Sequence Length:</strong> {hidden_data['sequence_length']}</p>"
    
    html += "<p><strong>Layer-wise Statistics:</strong></p>"
    html += "<table style='width: 100%; border-collapse: collapse;'>"
    html += "<tr style='background: #e0e0e0;'>"
    html += "<th style='padding: 8px; text-align: left;'>Layer</th>"
    html += "<th style='padding: 8px; text-align: right;'>Mean</th>"
    html += "<th style='padding: 8px; text-align: right;'>Std</th>"
    html += "<th style='padding: 8px; text-align: right;'>Norm</th>"
    html += "</tr>"
    
    for stats in hidden_data['hidden_states_stats'][:5]:  # Show first 5 layers
        html += "<tr style='border-bottom: 1px solid #ccc;'>"
        html += f"<td style='padding: 8px;'>Layer {stats['layer']}</td>"
        html += f"<td style='padding: 8px; text-align: right;'>{stats['mean']:.4f}</td>"
        html += f"<td style='padding: 8px; text-align: right;'>{stats['std']:.4f}</td>"
        html += f"<td style='padding: 8px; text-align: right;'>{stats['norm']:.2f}</td>"
        html += "</tr>"
    
    if hidden_data['num_layers'] > 5:
        html += f"<tr><td colspan='4' style='padding: 8px; text-align: center; color: #666;'>"
        html += f"... and {hidden_data['num_layers'] - 5} more layers</td></tr>"
    
    html += "</table>"
    html += "</div>"
    return html


def format_gradient_analysis(gradient_data: Dict[str, Any]) -> str:
    """
    Format gradient flow analysis for display.
    
    Args:
        gradient_data: Output from analyze_gradient_flow()
        
    Returns:
        Formatted HTML string
    """
    if 'error' in gradient_data:
        return f"<div style='color: red;'>Error: {gradient_data['error']}</div>"
    
    html = "<div style='background: #f0f2f6; padding: 15px; border-radius: 10px;'>"
    html += "<h4>📊 Gradient Flow Analysis</h4>"
    
    # Warnings
    if gradient_data['vanishing_gradient_detected']:
        html += "<div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
        html += "⚠️ <strong>Warning:</strong> Vanishing gradients detected (< 1e-7)"
        html += "</div>"
    
    if gradient_data['exploding_gradient_detected']:
        html += "<div style='background: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
        html += "⚠️ <strong>Warning:</strong> Exploding gradients detected (> 1e3)"
        html += "</div>"
    
    # Summary stats
    html += f"<p><strong>Total Parameters:</strong> {gradient_data['total_parameters']}</p>"
    html += f"<p><strong>Max Gradient:</strong> {gradient_data['max_gradient']:.6f}</p>"
    html += f"<p><strong>Min Gradient:</strong> {gradient_data['min_gradient']:.6f}</p>"
    html += f"<p><strong>Mean Gradient:</strong> {gradient_data['mean_gradient']:.6f}</p>"
    
    # Top gradients
    html += "<p><strong>Largest Gradients (Top 10):</strong></p>"
    html += "<div style='font-family: monospace; font-size: 0.85em; background: white; padding: 10px; border-radius: 5px; overflow-x: auto;'>"
    for grad_info in gradient_data['gradient_norms'][:10]:
        param_name = grad_info['parameter'].split('.')[-1]  # Show only last part
        html += f"{param_name}: {grad_info['gradient_norm']:.6f}<br>"
    html += "</div>"
    
    html += "</div>"
    return html
