"""
Difficulty scoring algorithm for CEFR corpus entries.
Produces a score from 1–100 based on CEFR level, word complexity, and phonetic features.
"""

import re
from .config import CEFR_DIFFICULTY_RANGES

# Simple syllable counter (heuristic)
def _count_syllables(word):
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    # Remove trailing silent e
    word = re.sub(r'e$', '', word)
    # Count vowel groups
    vowel_groups = re.findall(r'[aeiouy]+', word)
    count = len(vowel_groups)
    return max(1, count)


def score_difficulty(text, cefr_level, entry_type='word'):
    """
    Calculate a difficulty score (1–100) for a corpus entry.
    
    Parameters:
        text: The word, phrase, or sentence text
        cefr_level: One of A1, A2, B1, B2, C1, C2
        entry_type: 'word', 'phrase', 'sentence', etc.
    
    Returns:
        int: difficulty score 1–100
    """
    lo, hi = CEFR_DIFFICULTY_RANGES.get(cefr_level, (30, 70))
    base = (lo + hi) / 2.0
    
    # Length factor
    words = text.split()
    word_count = len(words)
    
    if entry_type == 'word':
        # Single word: longer words are harder
        char_len = len(text.replace('-', '').replace(' ', ''))
        syllables = _count_syllables(text)
        length_bonus = min(10, (char_len - 4) * 0.8)  # bonus for chars > 4
        syllable_bonus = min(8, (syllables - 1) * 2.0)
        score = base + length_bonus + syllable_bonus
    elif entry_type in ('phrase', 'minimal_pair'):
        # Phrase: more words = harder
        length_bonus = min(10, (word_count - 2) * 1.5)
        avg_syllables = sum(_count_syllables(w) for w in words) / max(1, word_count)
        syllable_bonus = min(8, (avg_syllables - 1) * 2.0)
        score = base + length_bonus + syllable_bonus
    else:
        # Sentence, passage, dialogue, etc.
        length_bonus = min(12, (word_count - 5) * 0.5)
        avg_word_len = sum(len(w) for w in words) / max(1, word_count)
        complexity_bonus = min(8, (avg_word_len - 4) * 1.5)
        score = base + length_bonus + complexity_bonus
    
    # Add small random-like variation based on text hash
    hash_variation = (hash(text) % 11) - 5  # -5 to +5
    score += hash_variation
    
    # Clamp to valid range
    return max(1, min(100, int(round(score))))
