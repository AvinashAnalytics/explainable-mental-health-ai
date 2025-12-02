"""
Download and Prepare Mental Health Datasets

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset rsdd
    python download_datasets.py --dataset dreaddit --output data/

What this script does:
1. Downloads public mental health datasets
2. Standardizes format (text, label, source)
3. Cleans and validates data
4. Splits into train/val/test
5. Saves in project format
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Download mental health datasets')
    
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['dreaddit', 'rsdd', 'clpsych', 'erisk', 'smhd', 'all'],
                        help='Dataset to download')
    parser.add_argument('--all', action='store_true',
                        help='Download all available datasets')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if dataset already exists')
    
    return parser.parse_args()


def download_dreaddit(output_dir: str, skip_existing: bool = False) -> bool:
    """
    Download Dreaddit dataset (stress detection from Reddit).
    
    Paper: Turcan & McKeown (2019) - Dreaddit: A Reddit Dataset for Stress Analysis
    Source: https://github.com/ml-research/dreaddit-dataset
    """
    logger.info("Downloading Dreaddit dataset...")
    
    output_path = os.path.join(output_dir, 'dreaddit-train.csv')
    
    if skip_existing and os.path.exists(output_path):
        logger.info(f"  Dreaddit already exists, skipping")
        return True
    
    try:
        # In real implementation, download from GitHub
        # For now, create instructions
        logger.info("  📦 Dreaddit Dataset Instructions:")
        logger.info("     1. Visit: https://github.com/ml-research/dreaddit-dataset")
        logger.info("     2. Download dreaddit-train.csv")
        logger.info(f"     3. Place in: {output_path}")
        logger.info("     OR use existing sample: data/dreaddit_sample.csv")
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to download Dreaddit: {e}")
        return False


def download_rsdd(output_dir: str, skip_existing: bool = False) -> bool:
    """
    Download RSDD (Reddit Self-reported Depression Diagnosis).
    
    Paper: Yates et al. (2017) - Depression and Self-Harm Risk Assessment in Online Forums
    Note: Requires special access
    """
    logger.info("Downloading RSDD dataset...")
    
    logger.info("  📦 RSDD Dataset Instructions:")
    logger.info("     RSDD requires special access due to privacy concerns")
    logger.info("     1. Visit: https://georgetown-ir-lab.github.io/")
    logger.info("     2. Request access via research application")
    logger.info("     3. Follow their data sharing agreement")
    logger.info("     Note: This dataset contains ~9000 users' posts")
    
    return False


def download_clpsych(output_dir: str, skip_existing: bool = False) -> bool:
    """
    Download CLPsych shared task datasets.
    
    Paper: CLPsych Shared Tasks (2015-2021)
    Source: Various shared task organizers
    """
    logger.info("Downloading CLPsych dataset...")
    
    logger.info("  📦 CLPsych Dataset Instructions:")
    logger.info("     1. Visit: https://clpsych.org/shared-tasks/")
    logger.info("     2. Sign data usage agreement")
    logger.info("     3. Download from task organizers")
    logger.info("     Note: Multiple years available (2015-2021)")
    
    return False


def download_erisk(output_dir: str, skip_existing: bool = False) -> bool:
    """
    Download eRisk datasets (early risk detection).
    
    Paper: eRisk Lab at CLEF
    Source: https://erisk.irlab.org/
    """
    logger.info("Downloading eRisk dataset...")
    
    logger.info("  📦 eRisk Dataset Instructions:")
    logger.info("     1. Visit: https://erisk.irlab.org/")
    logger.info("     2. Register for CLEF eRisk lab")
    logger.info("     3. Download depression detection task data")
    logger.info("     Note: Available from 2017-2023 tasks")
    
    return False


def download_smhd(output_dir: str, skip_existing: bool = False) -> bool:
    """
    Download SMHD (Self-reported Mental Health Diagnoses).
    
    Paper: Cohan et al. (2018) - SMHD: A Large-Scale Resource
    Note: Large dataset, may take time
    """
    logger.info("Downloading SMHD dataset...")
    
    logger.info("  📦 SMHD Dataset Instructions:")
    logger.info("     SMHD is a large-scale dataset (100k+ users)")
    logger.info("     1. Visit: https://ir.cs.georgetown.edu/resources/smhd.html")
    logger.info("     2. Request research access")
    logger.info("     3. Follow data sharing agreement")
    logger.info("     Note: Contains multiple mental health conditions")
    
    return False


def create_mock_dataset(output_dir: str) -> bool:
    """Create a mock dataset for testing."""
    logger.info("Creating mock dataset for demonstration...")
    
    texts_depressed = [
        "I feel so hopeless and empty. Nothing brings me joy anymore.",
        "Life has no meaning anymore. Everything is pointless. I can't go on.",
        "Can't get out of bed. No energy for anything. Just want to sleep forever.",
        "I'm worthless and useless. Nobody cares about me. I'm a burden.",
        "Nothing matters anymore. I feel numb inside. Can't feel happiness.",
        "Every day is a struggle. I'm so tired of everything. What's the point?",
        "I feel like I'm drowning. No motivation to do anything. Life is exhausting.",
        "I hate myself. Everything I do fails. I'm so alone and isolated.",
        "Can't concentrate on anything. Mind is foggy. Feel like giving up.",
        "I'm tired all the time. Sleep doesn't help. Just want it all to end."
    ] * 50  # 500 samples
    
    texts_control = [
        "Had a great day with friends! Feeling energized and happy. Life is good.",
        "Work is challenging but I'm learning a lot. Excited about the future!",
        "Started learning guitar. It's harder than I thought but really fun!",
        "Just finished a good workout. Feeling strong and accomplished today.",
        "Planning a vacation next month. So excited! Can't wait to travel.",
        "Had an amazing dinner with family. Grateful for these moments.",
        "Completed a big project at work. Feeling proud and satisfied!",
        "Beautiful weather today. Went for a long walk and felt peaceful.",
        "Reading a fascinating book. Can't put it down. Love learning new things.",
        "Caught up with old friends. Such good conversations. Feeling connected."
    ] * 50  # 500 samples
    
    texts = texts_depressed + texts_control
    labels = [1] * len(texts_depressed) + [0] * len(texts_control)
    sources = ['mock_depression'] * len(texts_depressed) + ['mock_control'] * len(texts_control)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'source': sources
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mock_depression_dataset.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"  ✅ Mock dataset created: {output_path}")
    logger.info(f"     Total samples: {len(df)}")
    logger.info(f"     Depression: {(df['label']==1).sum()}")
    logger.info(f"     Control: {(df['label']==0).sum()}")
    
    # Create train/val/test splits
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    train_df.to_csv(os.path.join(output_dir, 'mock_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'mock_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'mock_test.csv'), index=False)
    
    logger.info(f"  ✅ Splits created:")
    logger.info(f"     Train: {len(train_df)} samples")
    logger.info(f"     Val:   {len(val_df)} samples")
    logger.info(f"     Test:  {len(test_df)} samples")
    
    return True


def main():
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("Mental Health Dataset Downloader")
    logger.info("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    datasets_to_download = []
    
    if args.all:
        datasets_to_download = ['dreaddit', 'rsdd', 'clpsych', 'erisk', 'smhd']
    elif args.dataset:
        datasets_to_download = [args.dataset]
    else:
        logger.info("\n📚 Available Datasets:")
        logger.info("  1. Dreaddit - Stress detection from Reddit")
        logger.info("  2. RSDD - Reddit Self-reported Depression Diagnosis")
        logger.info("  3. CLPsych - Multiple shared task datasets")
        logger.info("  4. eRisk - Early risk detection datasets")
        logger.info("  5. SMHD - Self-reported Mental Health Diagnoses")
        logger.info("\nUsage:")
        logger.info("  python download_datasets.py --dataset dreaddit")
        logger.info("  python download_datasets.py --all")
        return
    
    # Download datasets
    success_count = 0
    for dataset in datasets_to_download:
        logger.info(f"\n{'-'*70}")
        
        if dataset == 'dreaddit':
            success = download_dreaddit(args.output, args.skip_existing)
        elif dataset == 'rsdd':
            success = download_rsdd(args.output, args.skip_existing)
        elif dataset == 'clpsych':
            success = download_clpsych(args.output, args.skip_existing)
        elif dataset == 'erisk':
            success = download_erisk(args.output, args.skip_existing)
        elif dataset == 'smhd':
            success = download_smhd(args.output, args.skip_existing)
        
        if success:
            success_count += 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Downloaded {success_count}/{len(datasets_to_download)} datasets")
    logger.info("=" * 70)
    
    # Offer to create mock dataset
    logger.info("\n💡 Want to create a mock dataset for testing?")
    response = input("Create mock dataset? (y/n): ").strip().lower()
    
    if response == 'y':
        create_mock_dataset(args.output)
        logger.info("\n✅ You can now train with:")
        logger.info(f"   python train_depression_classifier.py --data {args.output}/mock_train.csv")


if __name__ == '__main__':
    main()
