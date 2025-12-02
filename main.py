import argparse
import logging
import pandas as pd
import os
from src.config.schema import AppConfig
from src.data.loaders import load_dreaddit
from src.models.classical import ClassicalTrainer
from src.models.llm_adapter import LLMAdapter
from src.explainability.attention import AttentionExplainer
from src.prompts.manager import PromptManager
from src.evaluation.safety import SafetyGuard

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Explainable Mental Health AI")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "llm_eval"], required=True)
    parser.add_argument("--text", type=str, help="Input text for inference")
    args = parser.parse_args()

    # 1. Load Config
    logger.info(f"Loading config from {args.config}")
    config = AppConfig.load(args.config)
    
    # 2. Initialize Components (no DatasetLoader needed for current modes)
    
    # 3. Execution Modes
    if args.mode == "train":
        logger.info("Starting Training Mode")
        # Load Data
        train_path = 'data/dreaddit-train.csv'
        
        if not os.path.exists(train_path):
            logger.warning(f"Train file {train_path} not found. Creating dummy data.")
            dummy_df = pd.DataFrame({
                'text': ["I feel great today.", "I am so sad and hopeless.", "Normal day.", "Depression hurts."],
                'label': [0, 1, 0, 1],
                'source': ['dummy']*4
            })
            trainer = ClassicalTrainer(config)
            trainer.train(dummy_df, dummy_df)
        else:
            dataset = load_dreaddit(train_path)
            df = pd.DataFrame({
                'text': dataset.texts,
                'label': dataset.labels,
                'source': dataset.sources
            })
            trainer = ClassicalTrainer(config)
            trainer.train(df)

    elif args.mode == "inference":
        logger.info("Starting Inference Mode")
        if not args.text:
            logger.error("Please provide --text for inference")
            return

        # Classical Inference
        trainer = ClassicalTrainer(config)
        # Note: In a real scenario, we'd load the saved model, not init fresh.
        # But ClassicalTrainer inits from backbone. We need a load method.
        # For this demo, we assume the user wants to run inference on the backbone or a saved path.
        # Let's assume backbone for zero-shot BERT (bad idea) or just run the pipeline structure.
        
        result = trainer.predict(args.text)
        logger.info(f"Classical Prediction: {result}")
        
        # Explainability
        explainer = AttentionExplainer()
        top_tokens = explainer.extract_top_tokens(trainer.model, trainer.tokenizer, args.text)
        logger.info(f"Top Attention Tokens: {top_tokens}")

    elif args.mode == "llm_eval":
        logger.info("Starting LLM Evaluation Mode")
        if not args.text:
            logger.error("Please provide --text")
            return
            
        llm = LLMAdapter()
        prompter = PromptManager()
        
        # Zero-shot
        prompt = prompter.build_prompt("zero_shot", args.text)
        response = llm.generate(prompt)
        
        # Safety Check
        is_crisis, keywords = SafetyGuard.check_crisis_keywords(args.text)
        if is_crisis:
            logger.warning(f"CRISIS DETECTED: {keywords}")
            print(SafetyGuard.MEDICAL_DISCLAIMER)
            
        logger.info(f"LLM Response: {response['content']}")

if __name__ == "__main__":
    main()
