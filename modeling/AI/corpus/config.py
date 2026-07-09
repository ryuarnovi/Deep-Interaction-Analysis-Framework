"""
Central configuration for the Oxford CEFR Master Corpus Generator.
"""

# ─────────────────────────────── Target Counts ───────────────────────────────

VOCAB_TARGETS = {'A1': 1000, 'A2': 1500, 'B1': 2000, 'B2': 2500, 'C1': 3000, 'C2': 4000}

PHRASE_TARGET = 5000

SENTENCE_TARGETS = {'A1': 500, 'A2': 700, 'B1': 1000, 'B2': 1200, 'C1': 1000, 'C2': 600}

DIALOGUE_TARGET = 2000

SPEAKING_PROMPT_TARGETS = {'A1': 200, 'A2': 200, 'B1': 250, 'B2': 250, 'C1': 150, 'C2': 150}

READING_PASSAGE_TARGET = 500

MINIMAL_PAIR_TARGET = 500

GRAMMAR_TARGET = 2000

ACADEMIC_TARGET = 3000
BUSINESS_TARGET = 3000
TECHNOLOGY_TARGET = 3000
HEALTHCARE_TARGET = 3000

# ─────────────────────────────── CEFR Levels ─────────────────────────────────

CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

CEFR_DIFFICULTY_RANGES = {
    'A1': (5, 20),
    'A2': (15, 35),
    'B1': (30, 50),
    'B2': (45, 65),
    'C1': (60, 80),
    'C2': (75, 100),
}

# ─────────────────────────────── Categories ──────────────────────────────────

CATEGORIES = [
    'Greetings', 'Family', 'Daily Life', 'Education', 'University',
    'Technology', 'Software Engineering', 'Artificial Intelligence', 'Data Science',
    'Business', 'Finance', 'Accounting', 'Banking',
    'Healthcare', 'Pharmacy',
    'Government', 'Administration', 'Legal',
    'Research', 'Academic Writing', 'Communication',
    'Leadership', 'Project Management', 'Marketing',
    'Environment', 'Science', 'Engineering', 'Psychology',
    'Social Media', 'Entrepreneurship', 'Economics',
    'Travel', 'Transportation', 'Tourism', 'Food and Beverage',
]

DIALOGUE_TOPICS = [
    'Daily Conversation', 'University', 'Technology', 'Healthcare',
    'Finance', 'Business', 'Travel', 'Research',
    'Academic Presentation', 'Project Discussion', 'Customer Service', 'Job Interview',
]

READING_TOPICS = [
    'Science', 'Technology', 'Business', 'Healthcare', 'Education',
    'Environment', 'AI', 'Research', 'Engineering', 'Finance',
]

GRAMMAR_TOPICS = [
    'Present Simple', 'Present Continuous', 'Present Perfect',
    'Past Simple', 'Past Continuous', 'Past Perfect',
    'Future Forms', 'Passive Voice', 'Conditionals',
    'Relative Clauses', 'Reported Speech', 'Modal Verbs', 'Academic Grammar',
]

# ─────────────────────────────── Audio Metadata ──────────────────────────────

GENDERS = ['Male', 'Female']
ACCENTS = ['Oxford English', 'British RP', 'Modern Standard British']
SPEEDS = ['Slow', 'Normal', 'Fast']
EMOTIONS = ['Neutral', 'Formal', 'Conversational']

AUDIO_SPEC = {
    'format': 'WAV',
    'sample_rate': 16000,
    'bit_depth': '16-bit PCM',
    'channels': 'Mono',
    'background_noise': 'Minimal',
}

# ─────────────────────────────── Type Labels ─────────────────────────────────

TYPES = [
    'word', 'phrase', 'sentence', 'dialogue', 'speaking_prompt',
    'reading_passage', 'minimal_pair', 'grammar',
    'academic', 'business', 'technology', 'healthcare',
]

# ─────────────────────────────── Part of Speech ──────────────────────────────

POS_TAGS = ['noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction',
            'pronoun', 'determiner', 'exclamation', 'number', 'modal verb',
            'auxiliary verb', 'phrasal verb']
