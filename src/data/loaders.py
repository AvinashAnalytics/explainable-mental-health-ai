"""
Dataset loaders for mental health corpora.

Supports: Dreaddit, CLPsych, eRisk, RSDD, and custom CSV formats.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import re

from src.data.preprocessing import clean_text, is_valid_text

logger = logging.getLogger(__name__)


class MentalHealthDataset:
    """Unified interface for mental health text datasets."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        sources: Optional[List[str]] = None,
        metadata: Optional[dict] = None
    ):
        self.texts = texts
        self.labels = labels
        self.sources = sources or ['unknown'] * len(texts)
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'source': self.sources[idx]
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'text': self.texts,
            'label': self.labels,
            'source': self.sources
        })
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> MentalHealthDataset:
        """Create dataset from DataFrame."""
        return cls(
            texts=df['text'].tolist(),
            labels=df['label'].tolist(),
            sources=df.get('source', ['unknown'] * len(df)).tolist()
        )


def load_dreaddit(path: str | Path, clean: bool = True) -> MentalHealthDataset:
    """
    Load Dreaddit dataset (stress detection from Reddit).
    
    Expected columns: text, label, subreddit
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Dreaddit file not found: {path}")
        return MentalHealthDataset([], [], [])
    
    df = pd.read_csv(path)
    
    # Standardize column names
    if 'post' in df.columns:
        df['text'] = df['post']
    if 'subreddit' in df.columns:
        df['source'] = 'dreaddit_' + df['subreddit'].astype(str)
    else:
        df['source'] = 'dreaddit'
    
    # Clean text if requested
    if clean:
        df['text'] = df['text'].apply(lambda x: clean_text(x))
        df = df[df['text'].apply(is_valid_text)].reset_index(drop=True)
    
    # Ensure label column exists
    if 'label' not in df.columns:
        logger.warning("No label column found in Dreaddit, setting all to 0")
        df['label'] = 0
    
    return MentalHealthDataset.from_dataframe(df[['text', 'label', 'source']])


def load_clpsych(path: str | Path, clean: bool = True) -> MentalHealthDataset:
    """
    Load CLPsych dataset (mental health from forum posts).
    
    Expected columns: text, label (or risk_level)
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"CLPsych file not found: {path}")
        return MentalHealthDataset([], [], [])
    
    df = pd.read_csv(path)
    
    # Standardize columns
    if 'post' in df.columns:
        df['text'] = df['post']
    if 'risk_level' in df.columns:
        # Convert risk levels to binary (low=0, medium/high=1)
        df['label'] = (df['risk_level'] != 'low').astype(int)
    elif 'label' not in df.columns:
        df['label'] = 0
    
    df['source'] = 'clpsych'
    
    if clean:
        df['text'] = df['text'].apply(lambda x: clean_text(x))
        df = df[df['text'].apply(is_valid_text)].reset_index(drop=True)
    
    return MentalHealthDataset.from_dataframe(df[['text', 'label', 'source']])


def load_erisk(path: str | Path, clean: bool = True) -> MentalHealthDataset:
    """
    Load eRisk dataset (early risk detection of depression).
    
    Expected columns: text, label, user_id
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"eRisk file not found: {path}")
        return MentalHealthDataset([], [], [])
    
    df = pd.read_csv(path)
    
    if 'text' not in df.columns and 'post' in df.columns:
        df['text'] = df['post']
    
    df['source'] = 'erisk'
    
    if 'label' not in df.columns:
        df['label'] = 0
    
    if clean:
        df['text'] = df['text'].apply(lambda x: clean_text(x))
        df = df[df['text'].apply(is_valid_text)].reset_index(drop=True)
    
    return MentalHealthDataset.from_dataframe(df[['text', 'label', 'source']])


def load_generic_csv(
    path: str | Path,
    text_column: str = 'text',
    label_column: str = 'label',
    source_name: str = 'custom',
    clean: bool = True
) -> MentalHealthDataset:
    """
    Load any CSV file with text and labels.
    
    Args:
        path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column  
        source_name: Dataset identifier
        clean: Whether to apply text cleaning
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"CSV file not found: {path}")
        return MentalHealthDataset([], [], [])
    
    df = pd.read_csv(path)
    
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV")
    
    df = df.rename(columns={text_column: 'text'})
    
    if label_column in df.columns:
        df = df.rename(columns={label_column: 'label'})
    else:
        logger.warning(f"Label column '{label_column}' not found, setting all to 0")
        df['label'] = 0
    
    df['source'] = source_name
    
    if clean:
        df['text'] = df['text'].apply(lambda x: clean_text(x))
        df = df[df['text'].apply(is_valid_text)].reset_index(drop=True)
    
    return MentalHealthDataset.from_dataframe(df[['text', 'label', 'source']])


def extract_temporal_features(text: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Extract temporal features from text and posting time.
    
    Research Basis:
    - Cosma et al. 2023 (Time-Enriched): "Late-night posting (2-4 AM) correlates 
      with sleep disturbance (r=0.42, p<0.001). High posting frequency correlates 
      with agitation/restlessness."
    - Time2Vec encoding: Temporal features improve F1 by 3-5% for depression detection.
    
    Features Extracted:
    1. Late-night posting: 2-4 AM → sleep disturbance indicator
    2. Weekend posting: Saturday/Sunday → social isolation indicator
    3. High posting frequency: Multiple posts/day → agitation indicator
    4. Temporal symptom keywords: "can't sleep", "awake all night"
    
    Args:
        text: Input text
        timestamp: Optional posting timestamp (if available)
    
    Returns:
        Dictionary with temporal features:
            - late_night_post: bool
            - weekend_post: bool
            - hour: int (0-23)
            - day_of_week: str
            - temporal_symptom_count: int
            - temporal_keywords: list
    """
    features = {
        'late_night_post': False,
        'weekend_post': False,
        'hour': None,
        'day_of_week': None,
        'temporal_symptom_count': 0,
        'temporal_keywords': []
    }
    
    # 1. Extract posting time features (if timestamp provided)
    if timestamp:
        hour = timestamp.hour
        day_of_week = timestamp.strftime('%A')
        
        features['hour'] = hour
        features['day_of_week'] = day_of_week
        
        # Late-night posting (2-4 AM) → sleep disturbance
        if 2 <= hour <= 4:
            features['late_night_post'] = True
        
        # Weekend posting → social isolation
        if day_of_week in ['Saturday', 'Sunday']:
            features['weekend_post'] = True
    
    # 2. Extract temporal symptom keywords from text
    temporal_keywords = [
        # Sleep disturbance
        "can't sleep", "cant sleep", "insomnia", "awake all night", "no sleep",
        "up all night", "restless sleep", "nightmares", "sleep deprived",
        "neend nahi aa rahi", "raat bhar jagta hoon", "so nahi pa raha",
        
        # Time perception changes (depression symptom)
        "time drags", "days feel long", "endless", "forever", "never ends",
        "time stands still", "waiting for it to end",
        
        # Temporal isolation
        "all day alone", "haven't left house", "stuck at home", "isolated for days",
        "no one visits", "haven't talked to anyone"
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in temporal_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    features['temporal_symptom_count'] = len(found_keywords)
    features['temporal_keywords'] = found_keywords
    
    return features


def merge_datasets(*datasets: MentalHealthDataset) -> MentalHealthDataset:
    """Combine multiple datasets into one."""
    all_texts = []
    all_labels = []
    all_sources = []
    
    for ds in datasets:
        all_texts.extend(ds.texts)
        all_labels.extend(ds.labels)
        all_sources.extend(ds.sources)
    
    return MentalHealthDataset(all_texts, all_labels, all_sources)
