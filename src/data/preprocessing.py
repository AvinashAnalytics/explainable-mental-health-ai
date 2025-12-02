import re
import html
from typing import Optional

class TextPreprocessor:
    """
    Handles text cleaning, normalization, and anonymization for mental health datasets.
    Follows guidelines from 2401.02984 (noise handling).
    """
    
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    
    @staticmethod
    def clean(text: Optional[str]) -> str:
        if not text or not isinstance(text, str):
            return ""
            
        # 1. HTML Unescape
        text = html.unescape(text)
        
        # 2. Remove URLs (Privacy & Noise)
        text = TextPreprocessor.URL_PATTERN.sub("", text)
        
        # 3. Anonymize Mentions (Privacy)
        text = TextPreprocessor.MENTION_PATTERN.sub("[USER]", text)
        
        # 4. Remove Hashtags (Keep content? Usually remove symbol, keep text. 
        # But for consistency with standard BERT inputs, we often remove them or split them.
        # Here we remove the symbol but keep the text is a common strategy, 
        # but let's stick to removing the whole tag if it's metadata-heavy, 
        # or just removing the # if it's integrated. 
        # Let's remove the # symbol only.)
        text = text.replace("#", "")
        
        # 5. Normalize Whitespace
        text = TextPreprocessor.WHITESPACE_PATTERN.sub(" ", text).strip()
        
        return text

    @staticmethod
    def is_valid_length(text: str, min_chars: int = 10) -> bool:
        return len(text) >= min_chars


# Convenience functions for backward compatibility
def clean_text(text: str) -> str:
    """Clean text using TextPreprocessor."""
    return TextPreprocessor.clean(text)


def is_valid_text(text: str, min_chars: int = 10) -> bool:
    """Check if text is valid length."""
    return TextPreprocessor.is_valid_length(text, min_chars)
