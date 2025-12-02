"""
Mental Health Content Filtering

Implements filtering strategies from research literature:
1. RSDD Pattern Filtering (Yates et al., 2017)
2. SMHD Pattern Filtering (Cohan et al., 2018)
3. Subreddit Filtering (Harrigian et al., 2020)

These filters remove explicit mental health discussions to prevent
topic leakage (posting "I'm depressed" in r/depression ≠ diagnosis).
"""

import re
from typing import List, Dict, Set

# Mental Health Subreddits (Harrigian et al., 2020 - 107 identified)
MH_SUBREDDITS = {
    # Depression-related
    'depression', 'depression_help', 'getting_over_it', 'depressed',
    'depression_memes', 'depression2', 'eood',
    
    # Anxiety-related  
    'anxiety', 'anxietyhelp', 'healthanxiety', 'socialanxiety',
    'panicdisorder', 'agoraphobia', 'panicattack', 'panicparty',
    
    # Suicide/Self-harm
    'suicidewatch', 'sw', 'selfharm', 'stopselfharm', 'madeofstysselast',
    
    # PTSD/Trauma
    'ptsd', 'cptsd', 'traumatoolbox', 'rapecounseling', 'survivorsofabuse',
    
    # Bipolar
    'bipolar', 'bipolarreddit', 'bipolar2', 'bipolarsos',
    
    # Eating Disorders
    'eatingdisorders', 'fuckeatingdisorders', 'eating_disorders',
    'anorexia', 'bulimia', 'bingeeating', 'edrecovery',
    
    # ADHD
    'adhd', 'adhdmemes', 'adhdwomen', 'twoxadhd', 'add', 'adhd_coaching',
    
    # OCD
    'ocd', 'ocpd', 'rocd', 'intrusivethoughts',
    
    # Schizophrenia/Psychosis
    'schizophrenia', 'schizoaffective', 'psychosis', 'psychoticreddit',
    
    # General Mental Health
    'mentalhealth', 'mentalhealthsupport', 'mentalillness',
    'bpd', 'borderlinepersonality', 'addiction', 'alcoholism',
    'anger', 'depersonalization', 'dp_dr', 'dpdr',
    'feelgood', 'hardshipmates', 'makemefeelbetter',
    'psychiatry', 'psychotherapy', 'therapyabuse',
    
    # Recovery/Support
    'getting_help', 'needafriend', 'kindvoice', 'mmmh',
    'traumatoolbox', 'mmfb'
}

# RSDD Language Patterns (Yates et al., 2017)
RSDD_MH_TERMS = [
    "adhd",
    "attention deficit hyperactivity disorder",
    "bi polar",
    "bi-polar",
    "bipolar",
    "depres",
    "depresion",
    "depression",
    "depressive",
    "diagnos",
    "mdd",
    "major depressive disorder",
    "mental disorder",
    "mental illness",
    "post traumatic stress disorder",
    "post-traumatic stress disorder",
    "posttraumatic stress disorder",
    "ptsd",
    "suffer from",
    "suffering from"
]

# RSDD Negative Patterns (to exclude false positives)
RSDD_NEG_PATTERNS = [
    "any help diagnos",
    "chooses not to be diagnos",
    "costs associated with diagnos",
    "diagnos her", "diagnos him", "diagnos people", "diagnos someone",
    "diagnostic analysis", "diagnostics", "diagnostic test",
    "help diagnos", "helped diagnos",
    "incorrectly diagnos", "misdiagnos", "never been diagnos",
    "not been diagnos", "not diagnosed",
    "refuse to be diagnos", "refused to be diagnos",
    "rule out diagnos", "self-diagnos", "to diagnos",
    "undiagnos", "was not diagnos", "who diagnosed"
]

# SMHD Language Patterns (Cohan et al., 2018) - Extensive list
SMHD_MH_TERMS = [
    'add', 'anxiety', 'add attention deficit disorder',
    'adhd', 'a.d.h.d.', 'ad/hd', 'adhd/add',
    'adhd (attention deficit hyperactivity disorder)',
    'affective bipolar disorder', 'alternating insanity',
    'anancastic neurosis', 'anancastic personality',
    'anorectic', 'anorexi', 'anorexia', 'anorexias', 'anorexic',
    'anxiety', 'anxiety disorder', 'anxiety disorder generalize',
    'anxiety disorder nos', 'anxietydisorderautism',
    'atypical depression', 'autis', 'autism',
    'autism spectrum disorder', 'autistic',
    'bi-polar', 'bi polar', 'binge eating',
    'binge eating disorder', 'binge-eating disorder',
    'bipolar', 'bipolar 1', 'bipolar 2', 'bipolar disorder',
    'bipolar i disorder', 'bipolar ii disorder',
    'borderline', 'borderline personality',
    'borderline personality disorder', 'bpd',
    'bulimia', 'bulimia nervosa', 'bulimic',
    'chronic depression', 'chronic major depression',
    'clinical depression', 'clinically depressed',
    'cyclothymia', 'cyclothymic disorder',
    'deep depression', 'depres', 'depresion',
    'depress', 'depressed', 'depression',
    'depressive disorder', 'dysthymia', 'dysthymic disorder',
    'eating disorder', 'eating disorder nos',
    'gad', 'generalized anxiety', 'generalized anxiety disorder',
    'major depression', 'major depressive disorder',
    'manic depression', 'manic-depressive',
    'obsessive compulsive', 'obsessive compulsive disorder',
    'ocd', 'panic', 'panic attack', 'panic disorder',
    'post traumatic stress', 'post-traumatic stress',
    'posttraumatic stress', 'post traumatic stress disorder',
    'post-traumatic stress disorder', 'posttraumatic stress disorder',
    'ptsd', 'ptsr', 'ptss',
    'recurrent depressive disorder',
    'schizoaffective', 'schizoaffective disorder',
    'schizophrenia', 'schizophrenia disorder', 'schizophrenic',
    'seasonal affective disorder', 'severe depression',
    'social anxiety', 'social anxiety disorder', 'social phobia'
]


class MentalHealthFilter:
    """
    Filter mental health content using research-validated patterns.
    
    Based on:
    - Harrigian et al. (2020): Subreddit filtering
    - Yates et al. (2017): RSDD pattern filtering
    - Cohan et al. (2018): SMHD pattern filtering
    """
    
    def __init__(
        self,
        filter_subreddits: bool = True,
        filter_method: str = None  # 'rsdd', 'smhd', or None
    ):
        """
        Initialize filter.
        
        Args:
            filter_subreddits: Whether to filter mental health subreddits
            filter_method: Pattern filtering method ('rsdd', 'smhd', None)
        """
        self.filter_subreddits = filter_subreddits
        self.filter_method = filter_method
        
        # Compile regex patterns for efficiency
        if filter_method == 'rsdd':
            self._compile_rsdd_patterns()
        elif filter_method == 'smhd':
            self._compile_smhd_patterns()
    
    def _compile_rsdd_patterns(self):
        """Compile RSDD regex patterns"""
        # Positive patterns (mental health terms)
        self.mh_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in RSDD_MH_TERMS) + r')\b',
            re.IGNORECASE
        )
        
        # Negative patterns (exclude false positives)
        self.neg_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in RSDD_NEG_PATTERNS) + r')\b',
            re.IGNORECASE
        )
    
    def _compile_smhd_patterns(self):
        """Compile SMHD regex patterns"""
        self.mh_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in SMHD_MH_TERMS) + r')\b',
            re.IGNORECASE
        )
    
    def should_filter_text(self, text: str) -> bool:
        """
        Check if text should be filtered.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text should be filtered (removed), False otherwise
        """
        if not self.filter_method:
            return False
        
        if self.filter_method == 'rsdd':
            # Must match MH term AND not match negative pattern
            has_mh_term = bool(self.mh_pattern.search(text))
            has_neg_pattern = bool(self.neg_pattern.search(text))
            return has_mh_term and not has_neg_pattern
        
        elif self.filter_method == 'smhd':
            # Simply check if text contains SMHD patterns
            return bool(self.mh_pattern.search(text))
        
        return False
    
    def should_filter_subreddit(self, subreddit: str) -> bool:
        """
        Check if subreddit should be filtered.
        
        Args:
            subreddit: Subreddit name (e.g., 'depression')
            
        Returns:
            True if posts from this subreddit should be filtered
        """
        if not self.filter_subreddits:
            return False
        
        return subreddit.lower() in MH_SUBREDDITS
    
    def filter_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Filter a list of posts.
        
        Args:
            posts: List of post dictionaries with 'text' and optionally 'subreddit'
            
        Returns:
            Filtered list of posts
        """
        filtered = []
        
        for post in posts:
            # Check subreddit filter
            if 'subreddit' in post and self.should_filter_subreddit(post['subreddit']):
                continue
            
            # Check text filter
            if 'text' in post and self.should_filter_text(post['text']):
                continue
            
            filtered.append(post)
        
        return filtered
    
    def get_filter_stats(self, posts: List[Dict]) -> Dict[str, int]:
        """
        Get statistics on filtering.
        
        Args:
            posts: Original list of posts
            
        Returns:
            Dictionary with filtering statistics
        """
        total = len(posts)
        filtered_subreddit = 0
        filtered_text = 0
        
        for post in posts:
            if 'subreddit' in post and self.should_filter_subreddit(post['subreddit']):
                filtered_subreddit += 1
                continue
            
            if 'text' in post and self.should_filter_text(post['text']):
                filtered_text += 1
        
        remaining = total - filtered_subreddit - filtered_text
        
        return {
            'total': total,
            'filtered_subreddit': filtered_subreddit,
            'filtered_text': filtered_text,
            'remaining': remaining,
            'filter_rate': (filtered_subreddit + filtered_text) / total if total > 0 else 0
        }


# Convenience functions
def filter_rsdd_posts(posts: List[Dict], filter_subreddits: bool = True) -> List[Dict]:
    """
    Filter posts using RSDD method.
    
    Args:
        posts: List of post dictionaries
        filter_subreddits: Whether to filter MH subreddits
        
    Returns:
        Filtered posts
    """
    filter_obj = MentalHealthFilter(
        filter_subreddits=filter_subreddits,
        filter_method='rsdd'
    )
    return filter_obj.filter_posts(posts)


def filter_smhd_posts(posts: List[Dict], filter_subreddits: bool = True) -> List[Dict]:
    """
    Filter posts using SMHD method.
    
    Args:
        posts: List of post dictionaries
        filter_subreddits: Whether to filter MH subreddits
        
    Returns:
        Filtered posts
    """
    filter_obj = MentalHealthFilter(
        filter_subreddits=filter_subreddits,
        filter_method='smhd'
    )
    return filter_obj.filter_posts(posts)


def is_mental_health_subreddit(subreddit: str) -> bool:
    """
    Check if subreddit is mental health related.
    
    Args:
        subreddit: Subreddit name
        
    Returns:
        True if mental health related
    """
    return subreddit.lower() in MH_SUBREDDITS


if __name__ == "__main__":
    # Example usage
    test_posts = [
        {
            'text': 'I was diagnosed with depression last year',
            'subreddit': 'depression'
        },
        {
            'text': 'Had a great day at work today!',
            'subreddit': 'CasualConversation'
        },
        {
            'text': 'Can anyone help diagnose my car issue?',
            'subreddit': 'MechanicAdvice'
        },
        {
            'text': 'I have ADHD and it affects my daily life',
            'subreddit': 'ADHD'
        }
    ]
    
    # Test RSDD filtering
    print("RSDD Filtering:")
    rsdd_filter = MentalHealthFilter(filter_subreddits=True, filter_method='rsdd')
    filtered = rsdd_filter.filter_posts(test_posts)
    stats = rsdd_filter.get_filter_stats(test_posts)
    
    print(f"Original posts: {stats['total']}")
    print(f"Filtered (subreddit): {stats['filtered_subreddit']}")
    print(f"Filtered (text): {stats['filtered_text']}")
    print(f"Remaining: {stats['remaining']}")
    print(f"\nRemaining posts:")
    for post in filtered:
        print(f"  - {post['text'][:50]}... (r/{post['subreddit']})")
