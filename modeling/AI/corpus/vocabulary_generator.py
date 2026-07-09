"""
Vocabulary Generator for the Oxford CEFR Master Corpus.
Generates 14,000+ vocabulary entries with IPA, definitions, examples, and difficulty scores.
"""

import os
import random
import pandas as pd
from .config import VOCAB_TARGETS, CATEGORIES, CEFR_LEVELS
from .ipa_engine import get_ipa
from .difficulty_scorer import score_difficulty

random.seed(42)

# ─────────────── Category assignment by word characteristics ─────────────────

CATEGORY_KEYWORDS = {
    'Greetings': ['hello', 'hi', 'goodbye', 'bye', 'welcome', 'greet', 'morning', 'evening', 'afternoon', 'hey', 'dear'],
    'Family': ['family', 'mother', 'father', 'sister', 'brother', 'parent', 'child', 'son', 'daughter', 'aunt', 'uncle', 'cousin', 'grandparent', 'grandmother', 'grandfather', 'husband', 'wife', 'baby', 'sibling', 'nephew', 'niece', 'relative', 'marriage', 'wedding', 'divorce'],
    'Daily Life': ['house', 'home', 'room', 'kitchen', 'bathroom', 'bedroom', 'door', 'window', 'garden', 'cook', 'eat', 'drink', 'sleep', 'wake', 'clean', 'wash', 'shop', 'buy', 'wear', 'clothes', 'shoe', 'dress', 'shirt'],
    'Education': ['school', 'student', 'teacher', 'learn', 'study', 'class', 'lesson', 'exam', 'test', 'grade', 'homework', 'book', 'read', 'write', 'pen', 'pencil', 'notebook', 'education', 'knowledge', 'curriculum', 'scholarship'],
    'University': ['university', 'college', 'professor', 'lecture', 'campus', 'degree', 'undergraduate', 'postgraduate', 'thesis', 'dissertation', 'semester', 'academic', 'faculty', 'dean', 'enrol'],
    'Technology': ['computer', 'internet', 'software', 'hardware', 'digital', 'device', 'screen', 'keyboard', 'mouse', 'network', 'online', 'website', 'app', 'download', 'upload', 'technology', 'electronic', 'virtual', 'cyber'],
    'Software Engineering': ['code', 'program', 'algorithm', 'database', 'debug', 'compile', 'deploy', 'framework', 'api', 'backend', 'frontend', 'server', 'client', 'repository', 'version'],
    'Artificial Intelligence': ['artificial', 'intelligence', 'machine', 'learning', 'neural', 'deep', 'model', 'training', 'prediction', 'classification', 'regression', 'automation', 'robot', 'cognitive'],
    'Data Science': ['data', 'analysis', 'statistics', 'visualization', 'dataset', 'correlation', 'regression', 'hypothesis', 'sample', 'variable', 'metric', 'insight', 'pattern', 'trend', 'forecast'],
    'Business': ['business', 'company', 'market', 'customer', 'client', 'profit', 'revenue', 'sale', 'trade', 'industry', 'enterprise', 'corporate', 'contract', 'negotiate', 'deal'],
    'Finance': ['finance', 'money', 'bank', 'invest', 'stock', 'bond', 'interest', 'loan', 'credit', 'debt', 'budget', 'tax', 'income', 'expense', 'asset', 'capital', 'dividend'],
    'Healthcare': ['health', 'hospital', 'doctor', 'nurse', 'patient', 'medicine', 'treatment', 'surgery', 'disease', 'symptom', 'diagnosis', 'therapy', 'clinic', 'prescription', 'vaccine', 'pharmacy'],
    'Science': ['science', 'experiment', 'research', 'theory', 'hypothesis', 'evidence', 'laboratory', 'chemical', 'physics', 'biology', 'molecule', 'atom', 'cell', 'gene', 'evolution'],
    'Environment': ['environment', 'climate', 'pollution', 'recycle', 'sustainable', 'ecology', 'conservation', 'emission', 'carbon', 'renewable', 'biodiversity', 'deforestation', 'ecosystem'],
    'Travel': ['travel', 'trip', 'journey', 'flight', 'airport', 'hotel', 'tourist', 'passport', 'luggage', 'destination', 'vacation', 'holiday', 'adventure', 'explore', 'guide'],
    'Food and Beverage': ['food', 'meal', 'restaurant', 'cook', 'recipe', 'ingredient', 'breakfast', 'lunch', 'dinner', 'snack', 'coffee', 'tea', 'juice', 'fruit', 'vegetable', 'meat', 'bread', 'cake', 'sugar', 'salt'],
    'Communication': ['speak', 'talk', 'listen', 'conversation', 'discuss', 'debate', 'argue', 'explain', 'present', 'communicate', 'language', 'speech', 'voice', 'pronunciation', 'accent'],
    'Legal': ['law', 'legal', 'court', 'judge', 'lawyer', 'crime', 'justice', 'right', 'regulation', 'legislation', 'contract', 'sue', 'guilty', 'innocent', 'verdict'],
    'Marketing': ['market', 'advertise', 'brand', 'campaign', 'promotion', 'consumer', 'audience', 'content', 'social', 'media', 'strategy', 'engagement', 'digital'],
}

# ─────────────── Definition templates by POS ─────────────────────────────────

DEF_TEMPLATES = {
    'n.': [
        'a person, thing, or concept related to {cat}',
        'an object or idea commonly associated with {cat}',
        'a term used in the context of {cat}',
    ],
    'v.': [
        'to perform an action related to {cat}',
        'to carry out or engage in an activity',
        'to do something in a particular way',
    ],
    'adj.': [
        'describing a quality or characteristic',
        'relating to or having the nature of something',
        'used to describe a particular state or condition',
    ],
    'adv.': [
        'in a particular manner or way',
        'to a certain degree or extent',
        'modifying the action or quality described',
    ],
    'prep.': [
        'indicating position, direction, or relationship',
        'used to show the connection between words',
    ],
    'conj.': [
        'used to connect words, phrases, or clauses',
        'a linking word in sentence construction',
    ],
    'pron.': [
        'a word used in place of a noun',
        'referring to a person, thing, or idea previously mentioned',
    ],
    'det.': [
        'a word that introduces a noun',
        'used before a noun to specify or identify it',
    ],
}

EXAMPLE_TEMPLATES = {
    'n.': [
        'The {word} was essential for the project.',
        'She studied the {word} carefully.',
        'Understanding {word} is important in this field.',
        'The {word} played a significant role.',
        'We need to consider the {word} in our analysis.',
    ],
    'v.': [
        'They decided to {word} the new approach.',
        'She managed to {word} the task efficiently.',
        'It is important to {word} before making decisions.',
        'The team will {word} the project next week.',
        'He learned to {word} through practice.',
    ],
    'adj.': [
        'The results were remarkably {word}.',
        'She found the experience quite {word}.',
        'The {word} approach proved effective.',
        'It was a {word} achievement for the team.',
        'The outcome was surprisingly {word}.',
    ],
    'adv.': [
        'She completed the work {word}.',
        'The system operates {word}.',
        'He {word} understood the concept.',
        'The project progressed {word}.',
        'They communicated {word} throughout.',
    ],
}

# ─────────────── Supplementary word lists per CEFR level ─────────────────────

SUPPLEMENTARY_WORDS = {
    'A1': [
        ('apple', 'n.'), ('bag', 'n.'), ('banana', 'n.'), ('bed', 'n.'), ('bird', 'n.'),
        ('blue', 'adj.'), ('box', 'n.'), ('bread', 'n.'), ('brown', 'adj.'), ('bus', 'n.'),
        ('cake', 'n.'), ('cap', 'n.'), ('cat', 'n.'), ('chair', 'n.'), ('cheese', 'n.'),
        ('chicken', 'n.'), ('clock', 'n.'), ('coat', 'n.'), ('cold', 'adj.'), ('cup', 'n.'),
        ('desk', 'n.'), ('dog', 'n.'), ('door', 'n.'), ('dress', 'n.'), ('egg', 'n.'),
        ('face', 'n.'), ('farm', 'n.'), ('fat', 'adj.'), ('finger', 'n.'), ('fish', 'n.'),
        ('flag', 'n.'), ('flat', 'adj.'), ('flower', 'n.'), ('foot', 'n.'), ('fork', 'n.'),
        ('fox', 'n.'), ('frog', 'n.'), ('fun', 'n.'), ('gate', 'n.'), ('gift', 'n.'),
        ('glad', 'adj.'), ('goat', 'n.'), ('gold', 'n.'), ('grape', 'n.'), ('gray', 'adj.'),
        ('gum', 'n.'), ('gym', 'n.'), ('hat', 'n.'), ('help', 'v.'), ('hill', 'n.'),
        ('horn', 'n.'), ('hug', 'v.'), ('ice', 'n.'), ('jam', 'n.'), ('jar', 'n.'),
        ('jet', 'n.'), ('jog', 'v.'), ('joy', 'n.'), ('jump', 'v.'), ('key', 'n.'),
        ('kid', 'n.'), ('king', 'n.'), ('kiss', 'v.'), ('kit', 'n.'), ('kite', 'n.'),
        ('lake', 'n.'), ('lamp', 'n.'), ('leaf', 'n.'), ('leg', 'n.'), ('lemon', 'n.'),
        ('lid', 'n.'), ('lip', 'n.'), ('log', 'n.'), ('map', 'n.'), ('mat', 'n.'),
        ('mug', 'n.'), ('nail', 'n.'), ('nest', 'n.'), ('net', 'n.'), ('nose', 'n.'),
        ('nut', 'n.'), ('oil', 'n.'), ('owl', 'n.'), ('pan', 'n.'), ('pea', 'n.'),
        ('pen', 'n.'), ('pet', 'n.'), ('pie', 'n.'), ('pig', 'n.'), ('pin', 'n.'),
        ('pink', 'adj.'), ('plum', 'n.'), ('pot', 'n.'), ('pup', 'n.'), ('queen', 'n.'),
        ('rain', 'n.'), ('rat', 'n.'), ('red', 'adj.'), ('rib', 'n.'), ('ring', 'n.'),
        ('rod', 'n.'), ('rug', 'n.'), ('run', 'v.'), ('sad', 'adj.'), ('sand', 'n.'),
        ('seed', 'n.'), ('ship', 'n.'), ('sick', 'adj.'), ('sing', 'v.'), ('sky', 'n.'),
        ('slip', 'v.'), ('snap', 'v.'), ('snow', 'n.'), ('soap', 'n.'), ('sock', 'n.'),
        ('spin', 'v.'), ('spoon', 'n.'), ('stem', 'n.'), ('step', 'n.'), ('stone', 'n.'),
        ('sun', 'n.'), ('swim', 'v.'), ('tail', 'n.'), ('tap', 'v.'), ('tent', 'n.'),
        ('thin', 'adj.'), ('tick', 'n.'), ('tie', 'n.'), ('tip', 'n.'), ('toe', 'n.'),
        ('tub', 'n.'), ('tug', 'v.'), ('van', 'n.'), ('vet', 'n.'), ('vine', 'n.'),
        ('wax', 'n.'), ('web', 'n.'), ('wet', 'adj.'), ('wig', 'n.'), ('win', 'v.'),
        ('wing', 'n.'), ('zip', 'n.'), ('zoo', 'n.'), ('bell', 'n.'), ('belt', 'n.'),
        ('bone', 'n.'), ('boot', 'n.'), ('bowl', 'n.'), ('brush', 'n.'), ('bulb', 'n.'),
        ('button', 'n.'), ('camp', 'n.'), ('card', 'n.'), ('cart', 'n.'), ('cave', 'n.'),
        ('cent', 'n.'), ('chain', 'n.'), ('chalk', 'n.'), ('chin', 'n.'), ('clay', 'n.'),
        ('cliff', 'n.'), ('cloud', 'n.'), ('coin', 'n.'), ('comb', 'n.'), ('corn', 'n.'),
        ('crab', 'n.'), ('crew', 'n.'), ('crop', 'n.'), ('crow', 'n.'), ('cube', 'n.'),
        ('curl', 'n.'), ('dam', 'n.'), ('dart', 'n.'), ('deer', 'n.'), ('dew', 'n.'),
        ('dip', 'v.'), ('dirt', 'n.'), ('dish', 'n.'), ('dock', 'n.'), ('dome', 'n.'),
        ('dot', 'n.'), ('dove', 'n.'), ('drum', 'n.'), ('duck', 'n.'), ('dump', 'n.'),
        ('dusk', 'n.'), ('dust', 'n.'), ('edge', 'n.'), ('elm', 'n.'), ('fair', 'adj.'),
    ],
    'A2': [
        ('ability', 'n.'), ('abroad', 'adv.'), ('accept', 'v.'), ('accident', 'n.'), ('achieve', 'v.'),
        ('active', 'adj.'), ('actual', 'adj.'), ('adventure', 'n.'), ('advice', 'n.'), ('afford', 'v.'),
        ('agriculture', 'n.'), ('ahead', 'adv.'), ('aim', 'n.'), ('alarm', 'n.'), ('album', 'n.'),
        ('alcohol', 'n.'), ('alive', 'adj.'), ('allow', 'v.'), ('amazing', 'adj.'), ('amount', 'n.'),
        ('ancient', 'adj.'), ('anger', 'n.'), ('announce', 'v.'), ('annual', 'adj.'), ('anxiety', 'n.'),
        ('apart', 'adv.'), ('appeal', 'n.'), ('appetite', 'n.'), ('arrange', 'v.'), ('arrow', 'n.'),
        ('attach', 'v.'), ('attract', 'v.'), ('audience', 'n.'), ('author', 'n.'), ('average', 'adj.'),
        ('awake', 'adj.'), ('aware', 'adj.'), ('awful', 'adj.'), ('balance', 'n.'), ('band', 'n.'),
        ('bargain', 'n.'), ('barrel', 'n.'), ('basis', 'n.'), ('battle', 'n.'), ('beam', 'n.'),
        ('benefit', 'n.'), ('billion', 'n.'), ('bitter', 'adj.'), ('blame', 'v.'), ('blank', 'adj.'),
        ('blast', 'n.'), ('blind', 'adj.'), ('block', 'n.'), ('blow', 'v.'), ('border', 'n.'),
        ('bother', 'v.'), ('bounce', 'v.'), ('bow', 'n.'), ('branch', 'n.'), ('brave', 'adj.'),
        ('breath', 'n.'), ('brick', 'n.'), ('brief', 'adj.'), ('broad', 'adj.'), ('bucket', 'n.'),
        ('bullet', 'n.'), ('bunch', 'n.'), ('burden', 'n.'), ('burn', 'v.'), ('burst', 'v.'),
        ('cabin', 'n.'), ('cable', 'n.'), ('calm', 'adj.'), ('capable', 'adj.'), ('capture', 'v.'),
        ('carbon', 'n.'), ('career', 'n.'), ('carpet', 'n.'), ('carve', 'v.'), ('casual', 'adj.'),
        ('cattle', 'n.'), ('caution', 'n.'), ('ceiling', 'n.'), ('celebrate', 'v.'), ('ceremony', 'n.'),
        ('champion', 'n.'), ('chapter', 'n.'), ('charm', 'n.'), ('chase', 'v.'), ('chest', 'n.'),
        ('circle', 'n.'), ('citizen', 'n.'), ('civil', 'adj.'), ('claim', 'v.'), ('classic', 'adj.'),
        ('clever', 'adj.'), ('climate', 'n.'), ('climb', 'v.'), ('clinic', 'n.'), ('cloth', 'n.'),
        ('clue', 'n.'), ('coach', 'n.'), ('coast', 'n.'), ('code', 'n.'), ('collapse', 'v.'),
        ('column', 'n.'), ('combine', 'v.'), ('comfort', 'n.'), ('command', 'n.'), ('comment', 'n.'),
        ('commit', 'v.'), ('companion', 'n.'), ('compete', 'v.'), ('complaint', 'n.'), ('complex', 'adj.'),
        ('compose', 'v.'), ('concentrate', 'v.'), ('concept', 'n.'), ('concern', 'n.'), ('conclusion', 'n.'),
        ('condition', 'n.'), ('conduct', 'v.'), ('confuse', 'v.'), ('connect', 'v.'), ('conscious', 'adj.'),
        ('consequence', 'n.'), ('consider', 'v.'), ('consist', 'v.'), ('constant', 'adj.'), ('construct', 'v.'),
        ('consume', 'v.'), ('contact', 'n.'), ('contain', 'v.'), ('content', 'n.'), ('contest', 'n.'),
        ('context', 'n.'), ('contract', 'n.'), ('contribute', 'v.'), ('control', 'v.'), ('convenient', 'adj.'),
        ('convince', 'v.'), ('cooperate', 'v.'), ('core', 'n.'), ('corporate', 'adj.'), ('correct', 'adj.'),
        ('council', 'n.'), ('courage', 'n.'), ('crash', 'v.'), ('creation', 'n.'), ('creature', 'n.'),
        ('crisis', 'n.'), ('criterion', 'n.'), ('critical', 'adj.'), ('crowd', 'n.'), ('crucial', 'adj.'),
        ('cruel', 'adj.'), ('cultural', 'adj.'), ('cure', 'n.'), ('curious', 'adj.'), ('current', 'adj.'),
        ('curve', 'n.'), ('custom', 'n.'), ('cycle', 'n.'), ('damage', 'n.'), ('dare', 'v.'),
        ('dawn', 'n.'), ('debate', 'n.'), ('decade', 'n.'), ('declare', 'v.'), ('decline', 'v.'),
        ('defeat', 'v.'), ('defend', 'v.'), ('define', 'v.'), ('degree', 'n.'), ('delay', 'n.'),
        ('deliver', 'v.'), ('demand', 'n.'), ('demonstrate', 'v.'), ('deny', 'v.'), ('depart', 'v.'),
        ('depend', 'v.'), ('deposit', 'n.'), ('depressed', 'adj.'), ('derive', 'v.'), ('deserve', 'v.'),
        ('desire', 'n.'), ('desperate', 'adj.'), ('destination', 'n.'), ('destroy', 'v.'), ('detect', 'v.'),
        ('determine', 'v.'), ('device', 'n.'), ('devote', 'v.'), ('diagram', 'n.'), ('dialogue', 'n.'),
    ],
    'B1': [
        ('abandon', 'v.'), ('abstract', 'adj.'), ('abundant', 'adj.'), ('accelerate', 'v.'), ('acceptance', 'n.'),
        ('accessible', 'adj.'), ('accomplish', 'v.'), ('accountable', 'adj.'), ('accumulate', 'v.'), ('accurate', 'adj.'),
        ('acquisition', 'n.'), ('activate', 'v.'), ('adaptation', 'n.'), ('adequate', 'adj.'), ('administer', 'v.'),
        ('adolescent', 'n.'), ('advocate', 'v.'), ('aesthetic', 'adj.'), ('affection', 'n.'), ('affordable', 'adj.'),
        ('aggregate', 'v.'), ('aggressive', 'adj.'), ('alignment', 'n.'), ('allocate', 'v.'), ('alteration', 'n.'),
        ('ambition', 'n.'), ('amendment', 'n.'), ('analogy', 'n.'), ('anniversary', 'n.'), ('anticipate', 'v.'),
        ('apparatus', 'n.'), ('appreciate', 'v.'), ('appropriate', 'adj.'), ('approximate', 'adj.'), ('arbitrary', 'adj.'),
        ('architect', 'n.'), ('arise', 'v.'), ('assembly', 'n.'), ('assertion', 'n.'), ('assign', 'v.'),
        ('assumption', 'n.'), ('assurance', 'n.'), ('atmosphere', 'n.'), ('attribute', 'n.'), ('authentic', 'adj.'),
        ('authority', 'n.'), ('autonomous', 'adj.'), ('awareness', 'n.'), ('backbone', 'n.'), ('bandwidth', 'n.'),
        ('bankrupt', 'adj.'), ('bargain', 'v.'), ('barrier', 'n.'), ('baseline', 'n.'), ('bearing', 'n.'),
        ('benchmark', 'n.'), ('beneficial', 'adj.'), ('beverage', 'n.'), ('blueprint', 'n.'), ('boundary', 'n.'),
        ('breakthrough', 'n.'), ('broadcast', 'v.'), ('bureaucracy', 'n.'), ('capability', 'n.'), ('capitalism', 'n.'),
        ('catalogue', 'n.'), ('category', 'n.'), ('certificate', 'n.'), ('challenge', 'n.'), ('chamber', 'n.'),
        ('champion', 'n.'), ('characteristic', 'n.'), ('circulation', 'n.'), ('circumstance', 'n.'), ('citizenship', 'n.'),
        ('classification', 'n.'), ('collaborate', 'v.'), ('commitment', 'n.'), ('commodity', 'n.'), ('compatible', 'adj.'),
        ('compensation', 'n.'), ('competence', 'n.'), ('competitive', 'adj.'), ('complement', 'n.'), ('compliance', 'n.'),
        ('comprehensive', 'adj.'), ('compromise', 'n.'), ('compulsory', 'adj.'), ('configuration', 'n.'), ('confirmation', 'n.'),
        ('confrontation', 'n.'), ('congestion', 'n.'), ('consciousness', 'n.'), ('conservation', 'n.'), ('consistency', 'n.'),
        ('consolidate', 'v.'), ('conspiracy', 'n.'), ('constituent', 'n.'), ('constraint', 'n.'), ('consultation', 'n.'),
        ('contemporary', 'adj.'), ('contradiction', 'n.'), ('controversy', 'n.'), ('convention', 'n.'), ('conversion', 'n.'),
        ('cooperation', 'n.'), ('coordination', 'n.'), ('correlation', 'n.'), ('correspondence', 'n.'), ('counselling', 'n.'),
        ('counterpart', 'n.'), ('coverage', 'n.'), ('creativity', 'n.'), ('credibility', 'n.'), ('criterion', 'n.'),
    ],
    'B2': [
        ('abolish', 'v.'), ('abstraction', 'n.'), ('accelerator', 'n.'), ('accumulation', 'n.'), ('acknowledgement', 'n.'),
        ('acquisition', 'n.'), ('activism', 'n.'), ('adaptation', 'n.'), ('adequacy', 'n.'), ('adhesion', 'n.'),
        ('adjective', 'n.'), ('adjudicate', 'v.'), ('admiration', 'n.'), ('adversary', 'n.'), ('affiliation', 'n.'),
        ('aggravate', 'v.'), ('agitation', 'n.'), ('allegiance', 'n.'), ('alleviate', 'v.'), ('amalgamation', 'n.'),
        ('ambiguity', 'n.'), ('amenity', 'n.'), ('amplification', 'n.'), ('analogue', 'n.'), ('anomaly', 'n.'),
        ('anthology', 'n.'), ('apparatus', 'n.'), ('apprehension', 'n.'), ('apprenticeship', 'n.'), ('arbitration', 'n.'),
        ('articulate', 'v.'), ('aspiration', 'n.'), ('assimilation', 'n.'), ('asymmetry', 'n.'), ('attribution', 'n.'),
        ('autonomy', 'n.'), ('backlash', 'n.'), ('benchmark', 'n.'), ('beneficiary', 'n.'), ('bilateral', 'adj.'),
        ('biodegradable', 'adj.'), ('biotechnology', 'n.'), ('blueprint', 'n.'), ('bottleneck', 'n.'), ('breakthrough', 'n.'),
        ('bureaucratic', 'adj.'), ('calibration', 'n.'), ('capitalism', 'n.'), ('categorisation', 'n.'), ('causality', 'n.'),
        ('centralisation', 'n.'), ('certification', 'n.'), ('chronological', 'adj.'), ('circumference', 'n.'), ('coalition', 'n.'),
        ('codification', 'n.'), ('coexistence', 'n.'), ('collaboration', 'n.'), ('commemoration', 'n.'), ('commissioner', 'n.'),
        ('communal', 'adj.'), ('compartment', 'n.'), ('compatibility', 'n.'), ('compilation', 'n.'), ('complication', 'n.'),
        ('comprehension', 'n.'), ('computation', 'n.'), ('conceivable', 'adj.'), ('concession', 'n.'), ('condensation', 'n.'),
        ('confederation', 'n.'), ('confrontation', 'n.'), ('congregation', 'n.'), ('conjunction', 'n.'), ('connotation', 'n.'),
        ('conscientious', 'adj.'), ('consolidation', 'n.'), ('constellation', 'n.'), ('constituency', 'n.'), ('consultation', 'n.'),
        ('contemplation', 'n.'), ('contingency', 'n.'), ('contraction', 'n.'), ('contradiction', 'n.'), ('convergence', 'n.'),
        ('conversion', 'n.'), ('coordination', 'n.'), ('correlation', 'n.'), ('correspondence', 'n.'), ('cultivation', 'n.'),
        ('culmination', 'n.'), ('customisation', 'n.'), ('decentralisation', 'n.'), ('decomposition', 'n.'), ('deduction', 'n.'),
        ('deficiency', 'n.'), ('degradation', 'n.'), ('deliberation', 'n.'), ('demographic', 'adj.'), ('denomination', 'n.'),
        ('depreciation', 'n.'), ('designation', 'n.'), ('deterioration', 'n.'), ('deviation', 'n.'), ('differentiation', 'n.'),
    ],
    'C1': [
        ('aberration', 'n.'), ('abolition', 'n.'), ('abrasion', 'n.'), ('abstinence', 'n.'), ('accolade', 'n.'),
        ('accreditation', 'n.'), ('acumen', 'n.'), ('adjudication', 'n.'), ('admonish', 'v.'), ('adversity', 'n.'),
        ('affidavit', 'n.'), ('aggrandise', 'v.'), ('allegory', 'n.'), ('amalgamate', 'v.'), ('ameliorate', 'v.'),
        ('amicable', 'adj.'), ('annexation', 'n.'), ('anomalous', 'adj.'), ('antecedent', 'n.'), ('aphorism', 'n.'),
        ('appeasement', 'n.'), ('arbitrariness', 'n.'), ('archaic', 'adj.'), ('articulation', 'n.'), ('ascertain', 'v.'),
        ('assiduous', 'adj.'), ('attenuation', 'n.'), ('austerity', 'n.'), ('authentication', 'n.'), ('authoritarian', 'adj.'),
        ('axiom', 'n.'), ('benevolence', 'n.'), ('bifurcation', 'n.'), ('bipartisan', 'adj.'), ('bolster', 'v.'),
        ('bourgeois', 'adj.'), ('brevity', 'n.'), ('bureaucratisation', 'n.'), ('calibrate', 'v.'), ('candid', 'adj.'),
        ('capitalise', 'v.'), ('caricature', 'n.'), ('categorical', 'adj.'), ('cessation', 'n.'), ('circumscribe', 'v.'),
        ('clandestine', 'adj.'), ('clemency', 'n.'), ('coalesce', 'v.'), ('coercion', 'n.'), ('cognisance', 'n.'),
        ('collateral', 'n.'), ('colloquial', 'adj.'), ('commemorate', 'v.'), ('commensurate', 'adj.'), ('commodify', 'v.'),
        ('compendium', 'n.'), ('compliant', 'adj.'), ('compounding', 'n.'), ('conciliatory', 'adj.'), ('concomitant', 'adj.'),
        ('confiscate', 'v.'), ('conglomerate', 'n.'), ('conjugate', 'v.'), ('connivance', 'n.'), ('conscription', 'n.'),
        ('consonance', 'n.'), ('contiguous', 'adj.'), ('contingent', 'adj.'), ('contravene', 'v.'), ('convalescence', 'n.'),
        ('corroborate', 'v.'), ('counterfeit', 'adj.'), ('covert', 'adj.'), ('credence', 'n.'), ('culminate', 'v.'),
        ('cursory', 'adj.'), ('curtail', 'v.'), ('cyberattack', 'n.'), ('debacle', 'n.'), ('decorum', 'n.'),
        ('deference', 'n.'), ('delineate', 'v.'), ('demarcation', 'n.'), ('denigrate', 'v.'), ('deprecate', 'v.'),
        ('dereliction', 'n.'), ('desiccate', 'v.'), ('despondent', 'adj.'), ('destitution', 'n.'), ('deter', 'v.'),
        ('diatribe', 'n.'), ('dichotomy', 'n.'), ('didactic', 'adj.'), ('digression', 'n.'), ('diminution', 'n.'),
        ('discrepancy', 'n.'), ('disseminate', 'v.'), ('dissolution', 'n.'), ('divergence', 'n.'), ('docile', 'adj.'),
        ('dogmatic', 'adj.'), ('dormant', 'adj.'), ('duplication', 'n.'), ('efficacy', 'n.'), ('egalitarian', 'adj.'),
    ],
    'C2': [
        ('abnegate', 'v.'), ('abrogate', 'v.'), ('abstemious', 'adj.'), ('acrimonious', 'adj.'), ('adjunct', 'n.'),
        ('adulterate', 'v.'), ('aggrandisement', 'n.'), ('alacrity', 'n.'), ('amorphous', 'adj.'), ('anachronism', 'n.'),
        ('anathema', 'n.'), ('antediluvian', 'adj.'), ('antipathy', 'n.'), ('apocryphal', 'adj.'), ('apotheosis', 'n.'),
        ('approbation', 'n.'), ('arcane', 'adj.'), ('asceticism', 'n.'), ('asperity', 'n.'), ('asseverate', 'v.'),
        ('atavism', 'n.'), ('attrition', 'n.'), ('avarice', 'n.'), ('bellicose', 'adj.'), ('beneficence', 'n.'),
        ('benighted', 'adj.'), ('blandishment', 'n.'), ('blithesome', 'adj.'), ('bombastic', 'adj.'), ('bourgeoisie', 'n.'),
        ('bucolic', 'adj.'), ('byzantine', 'adj.'), ('cabal', 'n.'), ('cacophony', 'n.'), ('calumny', 'n.'),
        ('capitulation', 'n.'), ('capricious', 'adj.'), ('castigate', 'v.'), ('caustic', 'adj.'), ('chicanery', 'n.'),
        ('circumlocution', 'n.'), ('clemency', 'n.'), ('cloister', 'n.'), ('coagulate', 'v.'), ('cognoscenti', 'n.'),
        ('colloquium', 'n.'), ('commencement', 'n.'), ('commiserate', 'v.'), ('commodious', 'adj.'), ('compunction', 'n.'),
        ('concatenation', 'n.'), ('conflagration', 'n.'), ('consanguinity', 'n.'), ('constriction', 'n.'), ('consummate', 'adj.'),
        ('contravention', 'n.'), ('contrite', 'adj.'), ('conundrum', 'n.'), ('convivial', 'adj.'), ('coruscate', 'v.'),
        ('countermand', 'v.'), ('cupidity', 'n.'), ('dearth', 'n.'), ('debilitate', 'v.'), ('decadence', 'n.'),
        ('defalcation', 'n.'), ('deification', 'n.'), ('deleterious', 'adj.'), ('demagogue', 'n.'), ('demeanour', 'n.'),
        ('denouement', 'n.'), ('deprecatory', 'adj.'), ('derelict', 'adj.'), ('desideratum', 'n.'), ('desultory', 'adj.'),
        ('devolve', 'v.'), ('dialectic', 'n.'), ('diaphanous', 'adj.'), ('diffidence', 'n.'), ('dilatory', 'adj.'),
        ('disabuse', 'v.'), ('discernment', 'n.'), ('discomfiture', 'n.'), ('discursive', 'adj.'), ('disingenuous', 'adj.'),
        ('disparage', 'v.'), ('dissemble', 'v.'), ('dissonance', 'n.'), ('distillation', 'n.'), ('doctrinaire', 'adj.'),
        ('duplicity', 'n.'), ('ebullience', 'n.'), ('edification', 'n.'), ('effacement', 'n.'), ('effervescent', 'adj.'),
        ('effulgence', 'n.'), ('egalitarianism', 'n.'), ('egregious', 'adj.'), ('elegy', 'n.'), ('elucidate', 'v.'),
        ('emancipation', 'n.'), ('embellishment', 'n.'), ('empiricism', 'n.'), ('encomium', 'n.'), ('endemic', 'adj.'),
        ('enervate', 'v.'), ('engender', 'v.'), ('enigmatic', 'adj.'), ('ennui', 'n.'), ('ephemeral', 'adj.'),
        ('epistemic', 'adj.'), ('equanimity', 'n.'), ('equivocal', 'adj.'), ('erudition', 'n.'), ('esoteric', 'adj.'),
        ('ethereal', 'adj.'), ('euphemism', 'n.'), ('exacerbate', 'v.'), ('excoriate', 'v.'), ('execrable', 'adj.'),
        ('exegesis', 'n.'), ('exemplary', 'adj.'), ('exhortation', 'n.'), ('existential', 'adj.'), ('expatriate', 'n.'),
        ('expediency', 'n.'), ('expiation', 'n.'), ('expostulate', 'v.'), ('extemporaneous', 'adj.'), ('extenuating', 'adj.'),
        ('extirpate', 'v.'), ('extraneous', 'adj.'), ('extrapolate', 'v.'), ('facetious', 'adj.'), ('factitious', 'adj.'),
        ('fastidious', 'adj.'), ('fatuous', 'adj.'), ('felicitous', 'adj.'), ('fervid', 'adj.'), ('fiduciary', 'adj.'),
        ('filibuster', 'n.'), ('flagrant', 'adj.'), ('florid', 'adj.'), ('forbearance', 'n.'), ('fortitude', 'n.'),
        ('fractious', 'adj.'), ('fulcrum', 'n.'), ('fulminate', 'v.'), ('garrulous', 'adj.'), ('genuflect', 'v.'),
        ('gerrymandering', 'n.'), ('grandiloquent', 'adj.'), ('gratuitous', 'adj.'), ('gregarious', 'adj.'), ('hagiography', 'n.'),
        ('hapless', 'adj.'), ('harbinger', 'n.'), ('hegemony', 'n.'), ('hermeneutics', 'n.'), ('heuristic', 'adj.'),
        ('histrionic', 'adj.'), ('homogeneity', 'n.'), ('hubris', 'n.'), ('iconoclast', 'n.'), ('idiosyncrasy', 'n.'),
        ('ignominious', 'adj.'), ('immutable', 'adj.'), ('impartial', 'adj.'), ('impecunious', 'adj.'), ('imperious', 'adj.'),
        ('imperturbable', 'adj.'), ('impervious', 'adj.'), ('impetuous', 'adj.'), ('implacable', 'adj.'), ('imprimatur', 'n.'),
        ('improvident', 'adj.'), ('impugn', 'v.'), ('inauspicious', 'adj.'), ('incandescent', 'adj.'), ('inchoate', 'adj.'),
        ('incipient', 'adj.'), ('incognito', 'adv.'), ('incongruous', 'adj.'), ('incontrovertible', 'adj.'), ('incredulity', 'n.'),
        ('indefatigable', 'adj.'), ('indeterminate', 'adj.'), ('indomitable', 'adj.'), ('ineffable', 'adj.'), ('inexorable', 'adj.'),
        ('infallible', 'adj.'), ('ingenuous', 'adj.'), ('inimical', 'adj.'), ('iniquitous', 'adj.'), ('inscrutable', 'adj.'),
        ('insidious', 'adj.'), ('insouciance', 'n.'), ('internecine', 'adj.'), ('intransigent', 'adj.'), ('introspection', 'n.'),
        ('inundate', 'v.'), ('invective', 'n.'), ('irascible', 'adj.'), ('irreverent', 'adj.'), ('itinerant', 'adj.'),
        ('jettison', 'v.'), ('jurisprudence', 'n.'), ('juxtaposition', 'n.'), ('labyrinthine', 'adj.'), ('laconic', 'adj.'),
        ('lambaste', 'v.'), ('lascivious', 'adj.'), ('laudable', 'adj.'), ('legerdemain', 'n.'), ('lethargic', 'adj.'),
        ('licentious', 'adj.'), ('limpid', 'adj.'), ('litigious', 'adj.'), ('logistical', 'adj.'), ('loquacious', 'adj.'),
        ('lugubrious', 'adj.'), ('machination', 'n.'), ('magnanimous', 'adj.'), ('malediction', 'n.'), ('maleficent', 'adj.'),
        ('malfeasance', 'n.'), ('malinger', 'v.'), ('malleable', 'adj.'), ('mandate', 'n.'), ('manifesto', 'n.'),
        ('maudlin', 'adj.'), ('mendacious', 'adj.'), ('mercurial', 'adj.'), ('meritocracy', 'n.'), ('metaphysical', 'adj.'),
        ('meticulous', 'adj.'), ('milieu', 'n.'), ('misanthrope', 'n.'), ('mitigate', 'v.'), ('moribund', 'adj.'),
        ('multifaceted', 'adj.'), ('munificent', 'adj.'), ('nascent', 'adj.'), ('nebulous', 'adj.'), ('nefarious', 'adj.'),
        ('nihilism', 'n.'), ('nomenclature', 'n.'), ('nonchalance', 'n.'), ('nonpareil', 'adj.'), ('obdurate', 'adj.'),
        ('obfuscate', 'v.'), ('obsequious', 'adj.'), ('obstreperous', 'adj.'), ('oeuvre', 'n.'), ('oligarchy', 'n.'),
        ('ominous', 'adj.'), ('omnipotent', 'adj.'), ('onerous', 'adj.'), ('opprobrium', 'n.'), ('oscillate', 'v.'),
        ('ostensible', 'adj.'), ('ostentatious', 'adj.'), ('palatable', 'adj.'), ('palindrome', 'n.'), ('panacea', 'n.'),
        ('paradigm', 'n.'), ('paradox', 'n.'), ('paragon', 'n.'), ('parochial', 'adj.'), ('parsimonious', 'adj.'),
        ('pastiche', 'n.'), ('paucity', 'n.'), ('pedagogy', 'n.'), ('pedantic', 'adj.'), ('penchant', 'n.'),
        ('penurious', 'adj.'), ('peremptory', 'adj.'), ('perfunctory', 'adj.'), ('pernicious', 'adj.'), ('perpetuate', 'v.'),
        ('perspicacious', 'adj.'), ('pertinacious', 'adj.'), ('philanthropy', 'n.'), ('philistine', 'n.'), ('phlegmatic', 'adj.'),
        ('platitude', 'n.'), ('plenary', 'adj.'), ('plenipotentiary', 'n.'), ('plutocracy', 'n.'), ('polemical', 'adj.'),
        ('pontificate', 'v.'), ('pragmatism', 'n.'), ('precarious', 'adj.'), ('precipitate', 'v.'), ('precocious', 'adj.'),
        ('predilection', 'n.'), ('preponderance', 'n.'), ('prerogative', 'n.'), ('prescient', 'adj.'), ('presumptuous', 'adj.'),
        ('prevaricate', 'v.'), ('probity', 'n.'), ('proclivity', 'n.'), ('prodigious', 'adj.'), ('profligate', 'adj.'),
        ('prognosticate', 'v.'), ('promulgate', 'v.'), ('propensity', 'n.'), ('propitious', 'adj.'), ('prosaic', 'adj.'),
        ('proscribe', 'v.'), ('proselytise', 'v.'), ('provenance', 'n.'), ('pugnacious', 'adj.'), ('pulchritude', 'n.'),
        ('punctilious', 'adj.'), ('pusillanimous', 'adj.'), ('quagmire', 'n.'), ('quandary', 'n.'), ('querulous', 'adj.'),
        ('quiescence', 'n.'), ('quintessential', 'adj.'), ('quixotic', 'adj.'), ('raconteur', 'n.'), ('rapprochement', 'n.'),
        ('ratiocination', 'n.'), ('recalcitrant', 'adj.'), ('recapitulate', 'v.'), ('recidivism', 'n.'), ('reciprocity', 'n.'),
        ('recompense', 'n.'), ('recondite', 'adj.'), ('redolent', 'adj.'), ('redux', 'adj.'), ('refulgent', 'adj.'),
        ('relinquish', 'v.'), ('remonstrate', 'v.'), ('remuneration', 'n.'), ('reparation', 'n.'), ('reprehensible', 'adj.'),
        ('reprobate', 'n.'), ('repudiate', 'v.'), ('requisite', 'adj.'), ('rescind', 'v.'), ('resplendent', 'adj.'),
        ('restitution', 'n.'), ('resurgence', 'n.'), ('reticent', 'adj.'), ('retribution', 'n.'), ('retrospection', 'n.'),
        ('revocation', 'n.'), ('rhetoric', 'n.'), ('sacrosanct', 'adj.'), ('sagacious', 'adj.'), ('salient', 'adj.'),
        ('sanguine', 'adj.'), ('sardonic', 'adj.'), ('scrupulous', 'adj.'), ('sedentary', 'adj.'), ('seminal', 'adj.'),
        ('serendipity', 'n.'), ('shibboleth', 'n.'), ('sinecure', 'n.'), ('soliloquy', 'n.'), ('somnolent', 'adj.'),
        ('sophistry', 'n.'), ('specious', 'adj.'), ('spurious', 'adj.'), ('squalid', 'adj.'), ('staid', 'adj.'),
        ('stigmatise', 'v.'), ('stoical', 'adj.'), ('stratagem', 'n.'), ('strident', 'adj.'), ('subjugate', 'v.'),
        ('sublime', 'adj.'), ('subterfuge', 'n.'), ('succinct', 'adj.'), ('superfluous', 'adj.'), ('supplant', 'v.'),
        ('supposition', 'n.'), ('surreptitious', 'adj.'), ('sycophant', 'n.'), ('taciturn', 'adj.'), ('tangential', 'adj.'),
        ('tantamount', 'adj.'), ('tautology', 'n.'), ('tempestuous', 'adj.'), ('tenacious', 'adj.'), ('tenet', 'n.'),
        ('tergiversation', 'n.'), ('terrestrial', 'adj.'), ('timorous', 'adj.'), ('torpor', 'n.'), ('totemic', 'adj.'),
        ('transcendence', 'n.'), ('transient', 'adj.'), ('trepidation', 'n.'), ('truculent', 'adj.'), ('tumultuous', 'adj.'),
        ('turpitude', 'n.'), ('ubiquitous', 'adj.'), ('umbrage', 'n.'), ('unconscionable', 'adj.'), ('unctuous', 'adj.'),
        ('unequivocal', 'adj.'), ('unfathomable', 'adj.'), ('unilateral', 'adj.'), ('unprecedented', 'adj.'), ('untenable', 'adj.'),
        ('unwieldy', 'adj.'), ('usurp', 'v.'), ('utilitarian', 'adj.'), ('vacillate', 'v.'), ('vainglorious', 'adj.'),
        ('valediction', 'n.'), ('vapid', 'adj.'), ('variegated', 'adj.'), ('vehement', 'adj.'), ('venal', 'adj.'),
        ('venerable', 'adj.'), ('veracious', 'adj.'), ('verbose', 'adj.'), ('vicarious', 'adj.'), ('vicissitude', 'n.'),
        ('vindicate', 'v.'), ('virulent', 'adj.'), ('visceral', 'adj.'), ('vitriolic', 'adj.'), ('vituperate', 'v.'),
        ('vociferous', 'adj.'), ('volition', 'n.'), ('voracious', 'adj.'), ('warranted', 'adj.'), ('whet', 'v.'),
        ('zealous', 'adj.'), ('zenith', 'n.'), ('zephyr', 'n.'),
    ],
}


def _assign_category(word):
    """Assign a category to a word based on keyword matching."""
    word_lower = word.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in word_lower or word_lower in kw:
                return cat
    # Default assignment based on hash for even distribution
    remaining_cats = [c for c in CATEGORIES if c not in ['Greetings', 'Family']]
    return remaining_cats[hash(word) % len(remaining_cats)]


def _generate_definition(word, pos, category):
    """Generate a template-based definition."""
    # Normalise POS to match template keys
    pos_key = pos.split(',')[0].strip().split('/')[0].strip()
    if pos_key not in DEF_TEMPLATES:
        pos_key = 'n.'
    
    templates = DEF_TEMPLATES[pos_key]
    template = templates[hash(word) % len(templates)]
    return template.format(cat=category.lower(), word=word)


def _generate_example(word, pos):
    """Generate a template-based example sentence."""
    pos_key = pos.split(',')[0].strip().split('/')[0].strip()
    if pos_key not in EXAMPLE_TEMPLATES:
        pos_key = 'n.'
    
    templates = EXAMPLE_TEMPLATES[pos_key]
    template = templates[hash(word + 'ex') % len(templates)]
    return template.format(word=word)


def generate_vocabulary(existing_vocab_path='data/oxford_vocabulary.csv'):
    """
    Generate the full vocabulary dataset (14,000+ entries).
    Uses existing Oxford vocabulary as base and supplements with curated lists.
    """
    print("Generating Vocabulary Dataset...")
    
    # Load existing vocabulary
    existing_words = set()
    rows = []
    
    if os.path.exists(existing_vocab_path):
        df_existing = pd.read_csv(existing_vocab_path)
        for _, row in df_existing.iterrows():
            word = str(row['word']).strip()
            cefr = str(row['cefr_level']).strip()
            pos = str(row.get('part_of_speech', 'n.')).strip()
            
            if word.lower() in existing_words:
                continue
            existing_words.add(word.lower())
            
            ipa_val, ipa_src = get_ipa(word)
            category = _assign_category(word)
            definition = _generate_definition(word, pos, category)
            example = _generate_example(word, pos)
            difficulty = score_difficulty(word, cefr, 'word')
            
            rows.append({
                'word': word,
                'ipa': ipa_val,
                'part_of_speech': pos if pos != 'nan' else 'n.',
                'cefr_level': cefr,
                'category': category,
                'definition': definition,
                'example_sentence': example,
                'difficulty_score': difficulty,
                'ipa_source': ipa_src,
            })
    
    print(f"  Loaded {len(rows)} words from existing vocabulary.")
    
    # Add supplementary words per level
    for level in CEFR_LEVELS:
        target = VOCAB_TARGETS[level]
        current_count = sum(1 for r in rows if r['cefr_level'] == level)
        needed = max(0, target - current_count)
        
        supplement = SUPPLEMENTARY_WORDS.get(level, [])
        added = 0
        for word, pos in supplement:
            if added >= needed:
                break
            if word.lower() in existing_words:
                continue
            existing_words.add(word.lower())
            
            ipa_val, ipa_src = get_ipa(word)
            category = _assign_category(word)
            definition = _generate_definition(word, pos, category)
            example = _generate_example(word, pos)
            difficulty = score_difficulty(word, level, 'word')
            
            rows.append({
                'word': word,
                'ipa': ipa_val,
                'part_of_speech': pos,
                'cefr_level': level,
                'category': category,
                'definition': definition,
                'example_sentence': example,
                'difficulty_score': difficulty,
                'ipa_source': ipa_src,
            })
            added += 1
        
        print(f"  {level}: {current_count} existing + {added} supplementary = {current_count + added} total")
    
    df = pd.DataFrame(rows)
    print(f"  Total vocabulary entries: {len(df)}")
    return df
