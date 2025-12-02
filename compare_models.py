"""
Compare Multiple Models on Depression Detection

Usage:
    python compare_models.py --models models/trained/roberta_* models/trained/bert_*
    python compare_models.py --models models/trained/* --data data/test.csv
    python compare_models.py --benchmark

What this script does:
1. Loads multiple fine-tuned models
2. Evaluates all on same test set
3. Compares accuracy, F1, speed
4. Generates comparison report
5. Identifies best model
"""

import os
import sys
import argparse
import glob
import json
import time
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare depression detection models')
    
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Paths to model directories (supports wildcards)')
    parser.add_argument('--data', type=str, default='data/dreaddit-train.csv',
                        help='Test data CSV')
    parser.add_argument('--benchmark', action='store_true',
                        help='Use benchmark mode with predefined models')
    parser.add_argument('--test-size', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--output', type=str, default='outputs/model_comparison.json',
                        help='Output comparison report path')
    
    return parser.parse_args()


class ModelComparator:
    """Compare multiple depression detection models."""
    
    def __init__(self, model_paths: List[str], test_data: pd.DataFrame):
        self.model_paths = model_paths
        self.test_data = test_data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = []
    
    def evaluate_model(self, model_path: str) -> Dict:
        """Evaluate single model."""
        
        model_name = Path(model_path).name
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*70}")
        
        try:
            # Load model
            start_load = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
            model.eval()
            load_time = time.time() - start_load
            
            logger.info(f"  Loaded in {load_time:.2f}s")
            logger.info(f"  Parameters: {model.num_parameters():,}")
            
            # Tokenize data
            dataset = Dataset.from_pandas(self.test_data[['text', 'label']])
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=256
                )
            
            dataset = dataset.map(tokenize_function, batched=True)
            
            # Make predictions
            predictions = []
            inference_times = []
            
            logger.info(f"  Running inference on {len(dataset)} samples...")
            
            with torch.no_grad():
                for i in range(len(dataset)):
                    inputs = {
                        'input_ids': torch.tensor([dataset[i]['input_ids']]).to(self.device),
                        'attention_mask': torch.tensor([dataset[i]['attention_mask']]).to(self.device)
                    }
                    
                    start = time.time()
                    outputs = model(**inputs)
                    inference_times.append(time.time() - start)
                    
                    pred = torch.argmax(outputs.logits, dim=-1).item()
                    predictions.append(pred)
            
            labels = self.test_data['label'].values
            
            # Compute metrics
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='macro'
            )
            
            # Per-class metrics
            report = classification_report(
                labels, predictions, 
                target_names=['Control', 'Depression'],
                output_dict=True
            )
            
            result = {
                'model_name': model_name,
                'model_path': model_path,
                'parameters': model.num_parameters(),
                'load_time': load_time,
                'avg_inference_time': np.mean(inference_times),
                'total_inference_time': np.sum(inference_times),
                'samples_per_second': len(dataset) / np.sum(inference_times),
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                },
                'per_class_metrics': report
            }
            
            logger.info(f"\n  Results:")
            logger.info(f"    Accuracy:  {accuracy:.4f}")
            logger.info(f"    Precision: {precision:.4f}")
            logger.info(f"    Recall:    {recall:.4f}")
            logger.info(f"    F1 Score:  {f1:.4f}")
            logger.info(f"    Avg inference: {result['avg_inference_time']*1000:.2f}ms")
            logger.info(f"    Samples/sec: {result['samples_per_second']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"  Failed to evaluate {model_name}: {e}")
            return None
    
    def compare_all(self) -> List[Dict]:
        """Compare all models."""
        
        for model_path in self.model_paths:
            result = self.evaluate_model(model_path)
            if result:
                self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comparison report."""
        
        if not self.results:
            return {}
        
        # Sort by F1 score
        sorted_results = sorted(
            self.results, key=lambda x: x['metrics']['f1_score'], reverse=True
        )
        
        # Find best models
        best_accuracy = max(self.results, key=lambda x: x['metrics']['accuracy'])
        best_f1 = max(self.results, key=lambda x: x['metrics']['f1_score'])
        fastest = min(self.results, key=lambda x: x['avg_inference_time'])
        
        report = {
            'comparison_date': pd.Timestamp.now().isoformat(),
            'test_samples': len(self.test_data),
            'device': self.device,
            'models_compared': len(self.results),
            'rankings': {
                'by_f1_score': [
                    {
                        'rank': i + 1,
                        'model': r['model_name'],
                        'f1_score': r['metrics']['f1_score']
                    }
                    for i, r in enumerate(sorted_results)
                ],
                'by_accuracy': sorted(
                    [{'model': r['model_name'], 'accuracy': r['metrics']['accuracy']} 
                     for r in self.results],
                    key=lambda x: x['accuracy'], reverse=True
                ),
                'by_speed': sorted(
                    [{'model': r['model_name'], 'inference_ms': r['avg_inference_time']*1000} 
                     for r in self.results],
                    key=lambda x: x['inference_ms']
                )
            },
            'best_models': {
                'highest_accuracy': {
                    'model': best_accuracy['model_name'],
                    'accuracy': best_accuracy['metrics']['accuracy']
                },
                'highest_f1': {
                    'model': best_f1['model_name'],
                    'f1_score': best_f1['metrics']['f1_score']
                },
                'fastest_inference': {
                    'model': fastest['model_name'],
                    'inference_ms': fastest['avg_inference_time'] * 1000
                }
            },
            'detailed_results': self.results
        }
        
        return report
    
    def display_summary(self, report: Dict):
        """Display comparison summary."""
        
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\nModels Compared: {report['models_compared']}")
        print(f"Test Samples: {report['test_samples']}")
        print(f"Device: {report['device']}")
        
        print(f"\n🏆 Best Models:")
        print(f"  Highest Accuracy: {report['best_models']['highest_accuracy']['model']}")
        print(f"    → {report['best_models']['highest_accuracy']['accuracy']:.4f}")
        
        print(f"  Highest F1 Score: {report['best_models']['highest_f1']['model']}")
        print(f"    → {report['best_models']['highest_f1']['f1_score']:.4f}")
        
        print(f"  Fastest Inference: {report['best_models']['fastest_inference']['model']}")
        print(f"    → {report['best_models']['fastest_inference']['inference_ms']:.2f}ms")
        
        print(f"\n📈 Rankings by F1 Score:")
        for item in report['rankings']['by_f1_score']:
            print(f"  {item['rank']}. {item['model']:40s} F1: {item['f1_score']:.4f}")
        
        print("\n" + "="*70)


def main():
    args = parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Sample test set
    if len(df) > args.test_size:
        df = df.sample(n=args.test_size, random_state=42)
    
    logger.info(f"Test set: {len(df)} samples")
    logger.info(f"  Depression: {(df['label']==1).sum()}")
    logger.info(f"  Control: {(df['label']==0).sum()}")
    
    # Get model paths
    if args.benchmark:
        model_paths = glob.glob('models/trained/*')
    elif args.models:
        model_paths = []
        for pattern in args.models:
            model_paths.extend(glob.glob(pattern))
    else:
        logger.error("Provide --models or --benchmark")
        sys.exit(1)
    
    if not model_paths:
        logger.error("No models found")
        sys.exit(1)
    
    logger.info(f"\nFound {len(model_paths)} models to compare")
    
    # Compare models
    comparator = ModelComparator(model_paths, df)
    comparator.compare_all()
    
    # Generate report
    report = comparator.generate_report()
    
    if not report:
        logger.error("No models evaluated successfully")
        sys.exit(1)
    
    # Display summary
    comparator.display_summary(report)
    
    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📁 Full report saved to: {args.output}")
    
    # Recommendation
    best = report['best_models']['highest_f1']['model']
    logger.info(f"\n💡 Recommended model for production: {best}")


if __name__ == '__main__':
    main()
