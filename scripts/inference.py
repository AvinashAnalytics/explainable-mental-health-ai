"""
Main inference script for mental health text analysis.

Combines classical models, LLMs, and rule-based approaches with safety layer.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.config import get_config
from src.models.llm_adapter import MentalHealthLLM
from src.explainability.rule_explainer import explain_prediction
from src.safety.ethical_guard import SafetyGuard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MentalHealthAnalyzer:
    """
    Unified interface for mental health text analysis.
    
    Combines multiple analysis methods with safety checks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize analyzer with configuration.
        
        Args:
            config_path: Path to config YAML (optional)
        """
        # Load configuration
        if config_path:
            from src.core.config import Config
            self.config = Config.from_yaml(config_path)
        else:
            self.config = get_config()
        
        # Initialize components
        self.safety_guard = SafetyGuard(
            enable_crisis_routing=self.config.enable_crisis_routing,
            enable_filtering=True
        )
        
        # Initialize LLM (if configured)
        try:
            self.llm = MentalHealthLLM(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                templates_dir=self.config.get('llm.prompts_dir', 'config/prompts')
            )
            logger.info(f"LLM initialized: {self.config.llm_provider}/{self.config.llm_model}")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Rule-based analysis only.")
            self.llm = None
        
        logger.info("MentalHealthAnalyzer initialized")
    
    def analyze(
        self,
        text: str,
        method: str = 'hybrid',
        llm_prompt_type: str = 'zero_shot'
    ) -> Dict[str, Any]:
        """
        Analyze text for mental health indicators.
        
        Args:
            text: Input text to analyze
            method: Analysis method ('rule_based', 'llm', 'hybrid')
            llm_prompt_type: LLM prompting strategy (zero_shot, few_shot, cot, emotion_cot)
            
        Returns:
            Comprehensive analysis with symptoms, explanations, and safety info
        """
        if not text or not text.strip():
            return {
                'error': 'Empty input text',
                'text_length': 0
            }
        
        result = {
            'input_text': text,
            'input_length': len(text),
            'method': method,
            'analyses': {}
        }
        
        # Rule-based analysis (always run - fast and interpretable)
        try:
            rule_result = explain_prediction(text)
            result['analyses']['rule_based'] = rule_result
            logger.info(f"Rule-based: {rule_result.get('prediction', 'Unknown')} ({rule_result['symptom_count']} symptoms)")
        except Exception as e:
            logger.error(f"Rule-based analysis failed: {e}")
            result['analyses']['rule_based'] = {'error': str(e)}
        
        # LLM analysis (if requested and available)
        if method in ['llm', 'hybrid'] and self.llm:
            try:
                llm_result = self.llm.analyze(text, method=llm_prompt_type, include_safety=True)
                result['analyses']['llm'] = llm_result
                logger.info(f"LLM analysis: {llm_result.get('depression_likelihood', 'unknown')}")
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                result['analyses']['llm'] = {'error': str(e)}
        
        # Combine results for hybrid approach
        if method == 'hybrid':
            result['combined_assessment'] = self._combine_analyses(result['analyses'])
        
        # Apply safety layer
        result = self.safety_guard.process(result)
        
        return result
    
    def _combine_analyses(self, analyses: Dict) -> Dict[str, Any]:
        """
        Combine rule-based and LLM analyses into unified assessment.
        
        Uses rule-based for high-precision symptom detection and
        LLM for nuanced language understanding.
        """
        combined = {
            'method': 'hybrid',
            'confidence': 'medium'
        }
        
        # Get rule-based results
        rule = analyses.get('rule_based', {})
        llm = analyses.get('llm', {})
        
        # Severity from rule-based (more reliable)
        combined['severity'] = rule.get('severity', 'unknown')
        combined['symptom_count'] = rule.get('symptom_count', 0)
        combined['dsm5_symptoms'] = rule.get('detected_symptoms', [])
        
        # Explanation from LLM (more natural language)
        if 'explanation' in llm:
            combined['explanation'] = llm['explanation']
        elif 'explanation' in rule:
            combined['explanation'] = rule['explanation']
        else:
            combined['explanation'] = "Analysis completed."
        
        # Crisis flag (use OR logic - if either detects crisis)
        combined['requires_crisis_intervention'] = (
            rule.get('requires_crisis_intervention', False) or
            llm.get('requires_crisis_intervention', False)
        )
        
        # Confidence boost if both agree
        if rule.get('severity') == llm.get('depression_likelihood'):
            combined['confidence'] = 'high'
            combined['agreement'] = 'both_methods_agree'
        else:
            combined['confidence'] = 'medium'
            combined['agreement'] = 'methods_disagree'
        
        return combined
    
    def batch_analyze(
        self,
        texts: list[str],
        method: str = 'hybrid',
        output_file: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of input texts
            method: Analysis method
            output_file: Optional path to save results JSON
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing text {i+1}/{len(texts)}")
            try:
                result = self.analyze(text, method=method)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text {i+1}: {e}")
                results.append({'error': str(e), 'text_index': i})
        
        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """CLI interface for quick text analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mental Health Text Analysis')
    parser.add_argument('text', nargs='?', help='Text to analyze (or use --file)')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--method', choices=['rule_based', 'llm', 'hybrid'], default='hybrid')
    parser.add_argument('--llm-prompt', choices=['zero_shot', 'few_shot', 'cot', 'emotion_cot'], default='zero_shot')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--config', type=str, help='Config YAML file path')
    
    args = parser.parse_args()
    
    # Get input text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: Provide text via argument or --file")
        return
    
    # Run analysis
    analyzer = MentalHealthAnalyzer(config_path=args.config)
    result = analyzer.analyze(text, method=args.method, llm_prompt_type=args.llm_prompt)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
