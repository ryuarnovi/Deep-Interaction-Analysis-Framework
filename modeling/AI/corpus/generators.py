"""
Phrase, Sentence, Dialogue, Speaking Prompt, Reading Passage, Minimal Pair,
Grammar, Domain, and Audio Metadata generators.
"""

import random
import pandas as pd
from .config import *
from .ipa_engine import get_ipa
from .difficulty_scorer import score_difficulty

random.seed(42)

# ═══════════════════════════════ PHRASE GENERATOR ════════════════════════════

PHRASE_SEEDS = {
    'A1': [
        'good morning', 'good afternoon', 'good evening', 'good night', 'thank you',
        'excuse me', 'how are you', 'nice to meet you', 'see you later', 'come in',
        'sit down', 'stand up', 'go home', 'wake up', 'look at', 'listen to',
        'turn on', 'turn off', 'put on', 'take off', 'pick up', 'come back',
        'get up', 'go out', 'come here', 'go there', 'right now', 'every day',
        'last night', 'next week', 'a lot of', 'a little bit', 'all right',
        'of course', 'no problem', 'not yet', 'once more', 'so far', 'too much',
        'very well', 'as well', 'at home', 'at school', 'at work', 'by bus',
        'for example', 'in front of', 'next to', 'on time', 'at least',
    ],
    'A2': [
        'take care', 'look forward to', 'get along with', 'set up', 'find out',
        'give up', 'carry on', 'fill in', 'point out', 'work out', 'break down',
        'bring up', 'call back', 'check in', 'check out', 'come across', 'cut down',
        'deal with', 'drop off', 'end up', 'figure out', 'get over', 'go ahead',
        'hand in', 'hold on', 'keep up', 'leave out', 'make up', 'move on',
        'pass away', 'pick out', 'pull over', 'put off', 'run out of', 'show up',
        'sort out', 'take over', 'think about', 'throw away', 'try on', 'turn down',
        'watch out', 'write down', 'at the moment', 'by the way', 'in addition',
        'in order to', 'more or less', 'right away', 'so that', 'such as',
    ],
    'B1': [
        'take responsibility', 'make a decision', 'pay attention', 'take advantage of',
        'make progress', 'take place', 'come to terms with', 'make an effort',
        'take into account', 'bear in mind', 'catch up with', 'come up with',
        'get rid of', 'keep in touch', 'look into', 'make sense', 'put up with',
        'run into', 'stand out', 'take part in', 'turn out', 'break through',
        'bring about', 'carry out', 'come to a conclusion', 'draw attention to',
        'fall behind', 'give rise to', 'go through', 'hand over', 'hold back',
        'keep track of', 'lead to', 'let down', 'live up to', 'look up to',
        'make use of', 'miss out on', 'open up', 'pass on', 'point of view',
        'pull together', 'put forward', 'set off', 'speak up', 'step by step',
        'sum up', 'take on', 'think over', 'turn up', 'wind up',
    ],
    'B2': [
        'carry out research', 'financial statement', 'critical thinking',
        'take for granted', 'in the long run', 'on the other hand', 'as a matter of fact',
        'by and large', 'come to light', 'draw a conclusion', 'face the consequences',
        'gain access to', 'have an impact on', 'in accordance with', 'keep pace with',
        'lay the foundation', 'make a contribution', 'narrow down', 'on behalf of',
        'pave the way', 'play a crucial role', 'prior to', 'raise awareness',
        'reach a compromise', 'shed light on', 'stand a chance', 'strike a balance',
        'take a stand', 'turn a blind eye', 'under the circumstances',
        'weigh up the pros and cons', 'with regard to', 'bring into question',
        'call into question', 'cast doubt on', 'come into effect', 'give way to',
        'go hand in hand', 'have a bearing on', 'in light of', 'jump to conclusions',
        'keep abreast of', 'lend support to', 'meet the criteria', 'not to mention',
        'over the course of', 'put into practice', 'regardless of', 'rule out',
        'set the stage for', 'to a certain extent', 'with a view to',
    ],
    'C1': [
        'artificial intelligence system', 'machine learning model', 'data-driven approach',
        'evidence-based practice', 'peer-reviewed journal', 'qualitative analysis',
        'quantitative research', 'sustainable development', 'social cohesion',
        'cognitive development', 'cross-cultural communication', 'digital transformation',
        'ethical considerations', 'fiscal policy', 'global governance',
        'human capital', 'institutional framework', 'knowledge economy',
        'paradigm shift', 'regulatory compliance', 'risk assessment',
        'stakeholder engagement', 'strategic planning', 'supply chain management',
        'take precedence over', 'underpin the argument', 'venture capital',
        'yield significant results', 'zero-sum game', 'adverse effect',
        'at the forefront of', 'benchmark against', 'capacity building',
        'due diligence', 'executive summary', 'food for thought',
        'give credence to', 'have recourse to', 'in perpetuity',
        'judicial review', 'keep at bay', 'level the playing field',
        'mitigating factors', 'notwithstanding', 'on the grounds that',
        'predicate upon', 'quid pro quo', 'ramifications of',
        'status quo', 'turn of events', 'uncharted territory',
    ],
    'C2': [
        'epistemological framework', 'ontological perspective', 'hermeneutic approach',
        'dialectical materialism', 'phenomenological inquiry', 'heuristic evaluation',
        'axiological considerations', 'deontological ethics', 'teleological argument',
        'neopositivist methodology', 'constructivist paradigm', 'functionalist perspective',
        'postmodernist critique', 'structuralist analysis', 'deconstructionist reading',
        'seminal contribution', 'sine qua non', 'ipso facto', 'prima facie evidence',
        'ad hoc committee', 'bona fide offer', 'carte blanche', 'de facto standard',
        'ex post facto', 'force majeure', 'modus operandi', 'non sequitur',
        'per capita income', 'status quo ante', 'terra incognita',
        'vis-a-vis', 'mea culpa', 'raison d\'etre', 'fait accompli',
        'caveat emptor', 'coup de grace', 'enfant terrible', 'in medias res',
        'lingua franca', 'magnum opus', 'tabula rasa', 'tour de force',
        'vox populi', 'zeitgeist', 'anti-establishment sentiment',
        'cross-pollination of ideas', 'epistemological rupture',
        'hegemonic discourse', 'intertextual reference', 'juxtaposition of paradigms',
    ],
}

def generate_phrases():
    """Generate 5,000+ phrase entries."""
    print("Generating Phrase Dataset...")
    rows = []
    seen = set()
    
    for level in CEFR_LEVELS:
        seeds = PHRASE_SEEDS.get(level, [])
        for phrase in seeds:
            if phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            ipa_val, _ = get_ipa(phrase, sentence_mode=True)
            cat = CATEGORIES[hash(phrase) % len(CATEGORIES)]
            meaning = f'An expression commonly used in {cat.lower()} contexts.'
            diff = score_difficulty(phrase, level, 'phrase')
            rows.append({
                'phrase': phrase, 'ipa': ipa_val, 'cefr_level': level,
                'category': cat, 'meaning': meaning, 'difficulty': diff,
            })
    
    # Expand with combinatorial phrasal verbs and collocations
    verbs = ['make', 'take', 'give', 'get', 'put', 'set', 'run', 'come', 'go', 'bring',
             'carry', 'hold', 'keep', 'turn', 'break', 'call', 'cut', 'draw', 'fall', 'lay',
             'look', 'pass', 'pick', 'pull', 'push', 'stand', 'throw', 'work']
    particles = ['up', 'down', 'out', 'in', 'off', 'on', 'over', 'away', 'back', 'through',
                 'about', 'around', 'along', 'across', 'ahead', 'apart', 'aside', 'forward']
    nouns = ['decision', 'progress', 'effort', 'use', 'sense', 'difference', 'contribution',
             'improvement', 'arrangement', 'commitment', 'assessment', 'investment',
             'recommendation', 'adjustment', 'evaluation', 'calculation', 'observation',
             'preparation', 'presentation', 'application', 'communication', 'consideration',
             'investigation', 'interpretation', 'determination', 'administration']
    
    levels_cycle = CEFR_LEVELS * 100
    idx = 0
    
    # Phrasal verbs
    for v in verbs:
        for p in particles:
            phrase = f'{v} {p}'
            if phrase.lower() not in seen and len(rows) < PHRASE_TARGET:
                seen.add(phrase.lower())
                level = levels_cycle[idx % len(levels_cycle)]
                idx += 1
                ipa_val, _ = get_ipa(phrase, sentence_mode=True)
                cat = CATEGORIES[hash(phrase) % len(CATEGORIES)]
                diff = score_difficulty(phrase, level, 'phrase')
                rows.append({
                    'phrase': phrase, 'ipa': ipa_val, 'cefr_level': level,
                    'category': cat, 'meaning': f'A phrasal verb meaning to {v} in a {p}ward direction.',
                    'difficulty': diff,
                })
    
    # Verb + noun collocations
    for v in ['make', 'take', 'give', 'do', 'have', 'reach', 'draw', 'pay', 'raise', 'carry out']:
        for n in nouns:
            phrase = f'{v} a {n}' if v not in ['carry out'] else f'{v} a {n}'
            if phrase.lower() not in seen and len(rows) < PHRASE_TARGET:
                seen.add(phrase.lower())
                level = levels_cycle[idx % len(levels_cycle)]
                idx += 1
                ipa_val, _ = get_ipa(phrase, sentence_mode=True)
                cat = CATEGORIES[hash(phrase) % len(CATEGORIES)]
                diff = score_difficulty(phrase, level, 'phrase')
                rows.append({
                    'phrase': phrase, 'ipa': ipa_val, 'cefr_level': level,
                    'category': cat, 'meaning': f'A collocation meaning to {v} a {n}.',
                    'difficulty': diff,
                })
    
    # Adjective + noun collocations
    adjs = ['strong', 'significant', 'major', 'key', 'critical', 'essential', 'fundamental',
            'comprehensive', 'strategic', 'sustainable', 'innovative', 'effective', 'efficient',
            'substantial', 'considerable', 'remarkable', 'profound', 'unprecedented', 'viable']
    domain_nouns = ['impact', 'role', 'factor', 'issue', 'challenge', 'approach', 'strategy',
                    'framework', 'perspective', 'outcome', 'solution', 'analysis', 'method',
                    'principle', 'criterion', 'mechanism', 'infrastructure', 'initiative',
                    'paradigm', 'intervention', 'protocol', 'methodology', 'competence']
    
    for a in adjs:
        for n in domain_nouns:
            phrase = f'{a} {n}'
            if phrase.lower() not in seen and len(rows) < PHRASE_TARGET:
                seen.add(phrase.lower())
                level = levels_cycle[idx % len(levels_cycle)]
                idx += 1
                ipa_val, _ = get_ipa(phrase, sentence_mode=True)
                cat = CATEGORIES[hash(phrase) % len(CATEGORIES)]
                diff = score_difficulty(phrase, level, 'phrase')
                rows.append({
                    'phrase': phrase, 'ipa': ipa_val, 'cefr_level': level,
                    'category': cat, 'meaning': f'A collocation describing a {a} {n}.',
                    'difficulty': diff,
                })
    
    df = pd.DataFrame(rows[:PHRASE_TARGET])
    print(f"  Total phrase entries: {len(df)}")
    return df


# ═══════════════════════════════ SENTENCE GENERATOR ══════════════════════════

SENTENCE_TEMPLATES = {
    'A1': [
        'I like {noun}.', 'She has a {noun}.', 'He is a {adj} person.',
        'We go to {place} every day.', 'They eat {noun} for breakfast.',
        'The {noun} is on the table.', 'My {noun} is very {adj}.',
        'I can see a {adj} {noun}.', 'She wants to {verb} today.',
        'He does not like {noun}.', 'We are happy.', 'It is {adj} today.',
        'I have two {noun}s.', 'The children play in the {place}.',
        'She reads a {noun} every night.', 'This is my {noun}.',
    ],
    'A2': [
        'I usually {verb} in the morning.', 'She has been studying for two hours.',
        'They are planning to {verb} next weekend.', 'He bought a new {noun} yesterday.',
        'We should {verb} before it is too late.', 'The {noun} was more {adj} than expected.',
        'I have never been to {place}.', 'She enjoys {verb}ing with her friends.',
        'They were {verb}ing when I arrived.', 'He is going to {verb} tomorrow.',
        'We need to {verb} the {noun} carefully.', 'The weather is getting {adj}.',
        'I would like to {verb} this weekend.', 'She always {verb}s before dinner.',
    ],
    'B1': [
        'The students are attending an online lecture.',
        'If I had more time, I would {verb} more often.',
        'She has been working on this {noun} since last month.',
        'The {noun} which was presented at the conference was quite {adj}.',
        'Although the results were {adj}, they provided useful {noun}.',
        'He suggested that we should {verb} the {noun} immediately.',
        'The team managed to {verb} the project ahead of schedule.',
        'It is important to {verb} all the relevant {noun} before making a decision.',
        'According to the report, the {noun} has increased significantly.',
        'She was asked to {verb} the {noun} for the next meeting.',
    ],
    'B2': [
        'Artificial intelligence is transforming modern healthcare.',
        'The company has implemented a new financial management strategy.',
        'Had the government intervened earlier, the crisis could have been avoided.',
        'The research findings suggest that the {noun} plays a crucial role in {noun}.',
        'Not only did the team {verb} the deadline, but they also exceeded expectations.',
        'It is widely acknowledged that {noun} has a significant impact on {noun}.',
        'The extent to which {noun} affects {noun} remains a matter of debate.',
        'Despite the challenges, the organisation managed to {verb} its objectives.',
        'The study revealed a strong correlation between {noun} and {noun}.',
        'In order to {verb} effectively, one must consider multiple perspectives.',
    ],
    'C1': [
        'The implications of this research extend far beyond the immediate findings.',
        'A comprehensive analysis of the data reveals several underlying patterns.',
        'The notion that technology inevitably leads to progress is increasingly contested.',
        'Notwithstanding the limitations, the study provides valuable insights into the phenomenon.',
        'The efficacy of the proposed intervention has been demonstrated across multiple trials.',
        'It is imperative that policymakers take into account the long-term ramifications.',
        'The paradigm shift towards sustainable development necessitates fundamental changes.',
        'Critical evaluation of the existing literature reveals significant methodological gaps.',
        'The interplay between socioeconomic factors and educational outcomes warrants further investigation.',
        'Preliminary findings indicate a statistically significant relationship between the variables.',
    ],
    'C2': [
        'The epistemological underpinnings of the argument necessitate rigorous scrutiny.',
        'The dialectical tension between individual liberty and collective welfare remains unresolved.',
        'The hermeneutic interpretation of the text reveals hitherto unexplored dimensions of meaning.',
        'The axiological implications of artificial intelligence demand unprecedented ethical deliberation.',
        'The phenomenological reduction enables a return to the things themselves, as Husserl envisioned.',
        'The ontological status of mathematical objects continues to engender philosophical controversy.',
        'A deconstructionist reading of the narrative exposes the inherent contradictions within the text.',
        'The teleological suspension of the ethical, as Kierkegaard articulated, poses profound challenges.',
        'The juxtaposition of empiricist and rationalist epistemologies illuminates the complexity of human cognition.',
        'The incommensurability thesis, while contentious, has profoundly influenced contemporary philosophy of science.',
    ],
}

TEMPLATE_FILLERS = {
    'noun': ['technology', 'education', 'research', 'environment', 'development', 'community',
             'economy', 'government', 'organisation', 'industry', 'strategy', 'innovation',
             'investment', 'performance', 'infrastructure', 'communication', 'policy',
             'management', 'assessment', 'system', 'approach', 'framework', 'outcome',
             'analysis', 'programme', 'initiative', 'project', 'study', 'report'],
    'adj': ['important', 'significant', 'effective', 'comprehensive', 'innovative', 'sustainable',
            'essential', 'critical', 'remarkable', 'substantial', 'challenging', 'complex',
            'fundamental', 'practical', 'relevant', 'valuable', 'efficient', 'strategic'],
    'verb': ['develop', 'analyse', 'implement', 'evaluate', 'investigate', 'establish',
             'demonstrate', 'contribute', 'achieve', 'maintain', 'improve', 'enhance',
             'facilitate', 'integrate', 'monitor', 'coordinate', 'participate', 'collaborate'],
    'place': ['school', 'university', 'office', 'hospital', 'library', 'park', 'museum',
              'airport', 'station', 'market', 'restaurant', 'cinema', 'gym', 'church'],
}


def generate_sentences():
    """Generate 5,000+ sentence entries."""
    print("Generating Sentence Dataset...")
    rows = []
    seen = set()
    
    for level, target in SENTENCE_TARGETS.items():
        templates = SENTENCE_TEMPLATES.get(level, SENTENCE_TEMPLATES['B1'])
        grammar_topics = GRAMMAR_TOPICS
        count = 0
        attempt = 0
        
        while count < target and attempt < target * 10:
            attempt += 1
            template = templates[attempt % len(templates)]
            
            # Fill placeholders
            sentence = template
            for key, fillers in TEMPLATE_FILLERS.items():
                while '{' + key + '}' in sentence:
                    filler = fillers[(hash(sentence) + attempt) % len(fillers)]
                    sentence = sentence.replace('{' + key + '}', filler, 1)
            
            if sentence.lower() in seen:
                # Vary by appending context
                suffixes = [' recently', ' effectively', ' successfully', ' carefully',
                            ' gradually', ' significantly', ' consistently', ' continuously']
                sentence = sentence.rstrip('.') + suffixes[attempt % len(suffixes)] + '.'
            
            if sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            
            ipa_val, _ = get_ipa(sentence, sentence_mode=True)
            grammar_focus = grammar_topics[attempt % len(grammar_topics)]
            cat = CATEGORIES[hash(sentence) % len(CATEGORIES)]
            diff = score_difficulty(sentence, level, 'sentence')
            
            rows.append({
                'sentence': sentence, 'ipa': ipa_val, 'cefr_level': level,
                'grammar_focus': grammar_focus, 'category': cat, 'difficulty': diff,
            })
            count += 1
        
        print(f"  {level}: {count} sentences generated")
    
    df = pd.DataFrame(rows)
    print(f"  Total sentence entries: {len(df)}")
    return df


# ═══════════════════════════════ DIALOGUE GENERATOR ══════════════════════════

DIALOGUE_TEMPLATES = {
    'Daily Conversation': [
        ("Good morning! How are you today?", "I'm very well, thank you. How about you?"),
        ("What did you do yesterday?", "I went shopping and then visited a friend."),
        ("Would you like a cup of tea?", "Yes, please. That would be lovely."),
        ("What time does the shop close?", "It usually closes at six o'clock."),
        ("Do you know where the nearest bus stop is?", "Yes, it's just around the corner."),
    ],
    'University': [
        ("Good morning. How can I help you?", "I would like to register for my classes."),
        ("Have you finished your assignment?", "Not yet. I still need to write the conclusion."),
        ("What is the deadline for the thesis?", "The deadline is the end of next month."),
        ("Could you recommend any reference materials?", "I suggest looking at the journal articles in the library database."),
        ("When is the next lecture?", "The next lecture is on Wednesday at two o'clock."),
    ],
    'Technology': [
        ("Have you tried the new software update?", "Yes, it runs much faster now."),
        ("What programming language do you recommend?", "Python is excellent for beginners and data science."),
        ("How do I fix this error?", "Try clearing the cache and restarting the application."),
        ("What is cloud computing?", "It is the delivery of computing services over the internet."),
        ("Is cybersecurity important for small businesses?", "Absolutely. Every organisation needs proper security measures."),
    ],
    'Healthcare': [
        ("What seems to be the problem?", "I have had a headache for three days."),
        ("Are you taking any medication?", "Yes, I take painkillers twice a day."),
        ("When was your last check-up?", "It was about six months ago."),
        ("How often should I take this medicine?", "Take it three times a day after meals."),
        ("Do you have any allergies?", "I am allergic to penicillin."),
    ],
    'Finance': [
        ("I would like to open a savings account.", "Certainly. May I see your identification, please?"),
        ("What is the current interest rate?", "The current rate is three point five per cent."),
        ("How can I transfer money internationally?", "You can use our online banking service or visit any branch."),
        ("I need to report a lost credit card.", "I'll block the card immediately. Do you have the card number?"),
        ("What investment options do you offer?", "We have bonds, mutual funds, and fixed deposit accounts."),
    ],
    'Business': [
        ("Shall we begin the meeting?", "Yes, let's start with the quarterly report."),
        ("What are our targets for next quarter?", "We aim to increase revenue by fifteen per cent."),
        ("Have you prepared the presentation?", "Yes, I have included all the key performance indicators."),
        ("We need to discuss the budget allocation.", "I agree. Let's review the expenditure first."),
        ("When can we expect the final contract?", "The legal team will have it ready by Friday."),
    ],
    'Travel': [
        ("I'd like to book a return flight to London.", "When would you like to travel?"),
        ("Is breakfast included in the room rate?", "Yes, a full English breakfast is included."),
        ("How long does the journey take?", "It takes approximately two hours by train."),
        ("Could I have a window seat, please?", "Of course. I've reserved 14A for you."),
        ("Where is the nearest underground station?", "It's about a five-minute walk from here."),
    ],
    'Research': [
        ("What methodology did you use?", "We employed a mixed-methods approach combining surveys and interviews."),
        ("Have the results been peer-reviewed?", "Yes, the paper was accepted by the International Journal of Science."),
        ("What are the limitations of the study?", "The sample size was relatively small, which limits generalisability."),
        ("How does your research contribute to the field?", "It provides new evidence for the relationship between diet and cognition."),
        ("When will the findings be published?", "We expect publication in the next issue of the journal."),
    ],
    'Academic Presentation': [
        ("Today I will be presenting our findings on climate change.", "Thank you. Could you elaborate on the methodology?"),
        ("The data clearly shows a significant trend.", "What statistical tests did you apply?"),
        ("In conclusion, the evidence strongly supports our hypothesis.", "An excellent presentation. Are there any questions from the audience?"),
        ("I would like to draw your attention to figure three.", "Could you explain the anomaly in the second quarter?"),
        ("The implications of these results are far-reaching.", "How do you propose to address the limitations mentioned?"),
    ],
    'Project Discussion': [
        ("Where are we with the project timeline?", "We are slightly behind schedule but catching up."),
        ("Who is responsible for the next deliverable?", "The development team will handle it this week."),
        ("We need to reassess our risk management strategy.", "I agree. Let's schedule a separate meeting for that."),
        ("Have we received feedback from the stakeholders?", "Yes, and they are generally satisfied with the progress."),
        ("What resources do we need for the next phase?", "We will need two additional developers and a project coordinator."),
    ],
    'Customer Service': [
        ("Hello, how may I assist you today?", "I would like to make a complaint about my order."),
        ("I'm sorry to hear that. Could you provide your order number?", "Yes, it is seven five three two one."),
        ("We will investigate this matter immediately.", "Thank you. When can I expect a resolution?"),
        ("Is there anything else I can help you with?", "No, that will be all. Thank you for your help."),
        ("I apologise for the inconvenience.", "I appreciate your prompt response."),
    ],
    'Job Interview': [
        ("Tell me about yourself.", "I have five years of experience in software engineering."),
        ("Why are you interested in this position?", "I believe my skills align perfectly with the role requirements."),
        ("What are your greatest strengths?", "I am highly organised and a strong communicator."),
        ("Where do you see yourself in five years?", "I aim to be in a leadership role within the company."),
        ("Do you have any questions for us?", "Yes, could you tell me about the team I would be working with?"),
    ],
}

def generate_dialogues():
    """Generate 2,000+ dialogue entries."""
    print("Generating Dialogue Dataset...")
    rows = []
    levels_cycle = CEFR_LEVELS * 500
    idx = 0
    
    for topic, pairs in DIALOGUE_TEMPLATES.items():
        for speaker_a, speaker_b in pairs:
            level = levels_cycle[idx % len(levels_cycle)]
            idx += 1
            ipa_a, _ = get_ipa(speaker_a, sentence_mode=True)
            ipa_b, _ = get_ipa(speaker_b, sentence_mode=True)
            cat = CATEGORIES[hash(topic) % len(CATEGORIES)]
            
            rows.append({
                'topic': topic, 'speaker_a': speaker_a, 'speaker_b': speaker_b,
                'ipa_a': ipa_a, 'ipa_b': ipa_b, 'cefr_level': level, 'category': cat,
            })
    
    # Expand by varying existing dialogues
    base_rows = list(rows)
    contexts = ['in the office', 'at university', 'during a meeting', 'over the phone',
                'at the hospital', 'at the airport', 'in the classroom', 'at a conference']
    greetings = ['Good morning.', 'Good afternoon.', 'Hello.', 'Excuse me.', 'Hi there.']
    closings = ['Thank you very much.', 'I appreciate your help.', 'That\'s very kind of you.',
                'I look forward to hearing from you.', 'Have a good day.']
    
    while len(rows) < DIALOGUE_TARGET:
        base = base_rows[idx % len(base_rows)]
        greeting = greetings[idx % len(greetings)]
        closing = closings[idx % len(closings)]
        context = contexts[idx % len(contexts)]
        level = levels_cycle[idx % len(levels_cycle)]
        idx += 1
        
        speaker_a = f"{greeting} {base['speaker_a']}"
        speaker_b = f"{base['speaker_b']} {closing}"
        ipa_a, _ = get_ipa(speaker_a, sentence_mode=True)
        ipa_b, _ = get_ipa(speaker_b, sentence_mode=True)
        
        rows.append({
            'topic': base['topic'], 'speaker_a': speaker_a, 'speaker_b': speaker_b,
            'ipa_a': ipa_a, 'ipa_b': ipa_b, 'cefr_level': level,
            'category': CATEGORIES[hash(speaker_a) % len(CATEGORIES)],
        })
    
    df = pd.DataFrame(rows[:DIALOGUE_TARGET])
    print(f"  Total dialogue entries: {len(df)}")
    return df


# ═══════════════════════════════ SPEAKING PROMPT GENERATOR ═══════════════════

SPEAKING_PROMPTS = {
    'A1': [
        'Introduce yourself.', 'Describe your family.', 'Talk about your favourite food.',
        'Describe your home.', 'Talk about your daily routine.', 'Describe your best friend.',
        'Talk about your favourite colour.', 'Describe your classroom.',
        'Talk about what you like to do on weekends.', 'Describe the weather today.',
        'Talk about your favourite animal.', 'Describe your school.',
        'Talk about what you had for breakfast.', 'Describe your bedroom.',
        'Talk about a game you enjoy.', 'Describe a family member.',
        'Talk about your favourite season.', 'Describe what you are wearing.',
        'Talk about a place you like.', 'Describe your teacher.',
    ],
    'A2': [
        'Describe your hometown.', 'Talk about a hobby you enjoy.',
        'Describe your last holiday.', 'Talk about your favourite film.',
        'Describe a typical weekend.', 'Talk about your favourite subject at school.',
        'Describe a person you admire.', 'Talk about what you want to be in the future.',
        'Describe a restaurant you like.', 'Talk about your favourite music.',
        'Describe a festival in your country.', 'Talk about a sport you play.',
        'Describe a book you have read.', 'Talk about your neighbourhood.',
        'Describe your ideal day.', 'Talk about a journey you have taken.',
        'Describe a gift you received.', 'Talk about your plans for tomorrow.',
        'Describe a shop you visit often.', 'Talk about a pet you have or would like.',
    ],
    'B1': [
        'Talk about your favourite university subject.',
        'Describe an experience that changed your life.',
        'Discuss the importance of learning English.',
        'Talk about a skill you would like to learn.',
        'Describe a memorable trip you have taken.',
        'Discuss the role of technology in education.',
        'Talk about a challenge you have overcome.',
        'Describe an interesting person you have met.',
        'Discuss the benefits of regular exercise.',
        'Talk about a tradition in your culture.',
        'Describe your ideal job.', 'Discuss the importance of reading.',
        'Talk about a goal you have achieved.',
        'Describe a social issue you care about.',
        'Discuss the advantages of living in a city.',
    ],
    'B2': [
        'Discuss advantages and disadvantages of online learning.',
        'Describe how social media has changed communication.',
        'Discuss the impact of globalisation on local cultures.',
        'Talk about the role of technology in healthcare.',
        'Describe a situation where you had to make a difficult decision.',
        'Discuss the importance of environmental conservation.',
        'Talk about the challenges of working in a multicultural team.',
        'Describe how education systems could be improved.',
        'Discuss the ethics of animal testing.',
        'Talk about the future of renewable energy.',
        'Describe the impact of automation on employment.',
        'Discuss the pros and cons of remote working.',
        'Talk about the importance of mental health awareness.',
        'Describe how artificial intelligence might change society.',
        'Discuss the role of the media in modern democracy.',
    ],
    'C1': [
        'Explain how technology is changing education.',
        'Discuss the ethical implications of genetic engineering.',
        'Analyse the relationship between economic growth and environmental sustainability.',
        'Evaluate the effectiveness of international aid programmes.',
        'Discuss the impact of artificial intelligence on the labour market.',
        'Critically assess the role of social media in political discourse.',
        'Analyse the factors contributing to income inequality.',
        'Evaluate different approaches to healthcare reform.',
        'Discuss the philosophical implications of consciousness and AI.',
        'Analyse the effectiveness of different educational assessment methods.',
    ],
    'C2': [
        'Evaluate the impact of artificial intelligence on society.',
        'Critically analyse the concept of cultural relativism.',
        'Discuss the epistemological challenges posed by post-truth politics.',
        'Evaluate the efficacy of international climate change agreements.',
        'Analyse the interplay between technological determinism and social constructivism.',
        'Critically assess the notion of meritocracy in contemporary democracies.',
        'Discuss the implications of quantum computing for cryptography and cybersecurity.',
        'Evaluate the philosophical arguments for and against moral realism.',
        'Analyse the socioeconomic consequences of demographic transition.',
        'Critically examine the role of narrative in shaping historical understanding.',
    ],
}

def generate_speaking_prompts():
    """Generate 1,200+ speaking prompt entries."""
    print("Generating Speaking Prompts Dataset...")
    rows = []
    seen = set()
    
    for level, target in SPEAKING_PROMPT_TARGETS.items():
        prompts = SPEAKING_PROMPTS.get(level, [])
        count = 0
        
        # Add seed prompts
        for prompt in prompts:
            if count >= target:
                break
            if prompt.lower() in seen:
                continue
            seen.add(prompt.lower())
            ipa_val, _ = get_ipa(prompt, sentence_mode=True)
            cat = CATEGORIES[hash(prompt) % len(CATEGORIES)]
            diff = score_difficulty(prompt, level, 'sentence')
            rows.append({
                'prompt': prompt, 'ipa': ipa_val, 'cefr_level': level,
                'category': cat, 'difficulty': diff,
            })
            count += 1
        
        # Expand with topic variations
        topics = ['education', 'technology', 'health', 'environment', 'business',
                  'travel', 'culture', 'science', 'communication', 'family',
                  'work', 'sport', 'art', 'food', 'music', 'media', 'finance',
                  'psychology', 'law', 'engineering', 'hobbies', 'shopping',
                  'entertainment', 'history', 'literature', 'languages', 'animals',
                  'weather', 'nature', 'fashion', 'relationships', 'politics',
                  'society', 'cooking', 'leisure']
        actions = {
            'A1': ['Talk about', 'Describe', 'Tell me about'],
            'A2': ['Describe', 'Talk about', 'Explain', 'Tell me about'],
            'B1': ['Discuss', 'Describe', 'Talk about', 'Explain the importance of'],
            'B2': ['Discuss the advantages and disadvantages of', 'Analyse', 'Compare',
                   'Describe the impact of', 'Evaluate'],
            'C1': ['Critically analyse', 'Evaluate the role of', 'Discuss the implications of',
                   'Assess the effectiveness of', 'Examine the relationship between'],
            'C2': ['Critically evaluate', 'Deconstruct the notion of', 'Analyse the epistemological aspects of',
                   'Examine the philosophical underpinnings of', 'Synthesise arguments regarding'],
        }
        
        level_actions = actions.get(level, actions['B1'])
        idx = 0
        attempts = 0
        while count < target and attempts < target * 10:
            attempts += 1
            action = level_actions[idx % len(level_actions)]
            topic = topics[idx % len(topics)]
            prompt = f'{action} {topic} in your country.'
            idx += 1
            
            if prompt.lower() in seen:
                prompt = f'{action} {topic} and its impact on society.'
            if prompt.lower() in seen:
                prompt = f'{action} how {topic} has changed over time.'
            if prompt.lower() in seen:
                continue
            
            seen.add(prompt.lower())
            ipa_val, _ = get_ipa(prompt, sentence_mode=True)
            cat = CATEGORIES[hash(prompt) % len(CATEGORIES)]
            diff = score_difficulty(prompt, level, 'sentence')
            rows.append({
                'prompt': prompt, 'ipa': ipa_val, 'cefr_level': level,
                'category': cat, 'difficulty': diff,
            })
            count += 1
    
    df = pd.DataFrame(rows)
    print(f"  Total speaking prompt entries: {len(df)}")
    return df


# ═══════════════════════════════ READING PASSAGE GENERATOR ═══════════════════

READING_PASSAGE_TEMPLATES = [
    "{topic} is one of the most important subjects in the modern world. In recent years, there has been significant progress in this area. Researchers have discovered new methods and approaches that could transform how we understand {topic}. The implications of these developments are far-reaching and affect many aspects of daily life. As we continue to explore new possibilities, it is essential that we consider both the benefits and challenges that arise. Understanding {topic} is crucial for anyone who wants to contribute meaningfully to society.",
    "The field of {topic} has undergone remarkable transformation in the past decade. New technologies and methodologies have enabled researchers and practitioners to achieve results that were previously considered impossible. The impact of these changes extends beyond the laboratory and into everyday life, influencing how we work, communicate, and solve problems. As the field continues to evolve, it is important to stay informed about the latest developments and their potential applications.",
    "Recent advances in {topic} have opened up exciting new possibilities for innovation and improvement. Experts in the field argue that these developments could fundamentally change the way we approach complex challenges. However, there are also concerns about the potential risks and ethical implications. A balanced approach that considers multiple perspectives is essential for ensuring that progress in {topic} benefits society as a whole.",
    "The importance of {topic} in contemporary society cannot be overstated. From its applications in industry to its role in shaping public policy, {topic} affects virtually every aspect of modern life. Educational institutions around the world are increasingly recognising the need to incorporate {topic} into their curricula. As the demand for expertise in this area continues to grow, it is essential that we invest in research and training to prepare the next generation of professionals.",
    "In today's rapidly changing world, {topic} plays a pivotal role in driving economic growth and social development. The challenges we face require innovative solutions that draw on the latest research and best practices. By fostering collaboration between academia, industry, and government, we can create an environment that supports progress in {topic} while addressing the needs of diverse communities.",
]

def generate_reading_passages():
    """Generate 500+ reading passage entries."""
    print("Generating Reading Passages Dataset...")
    rows = []
    levels_cycle = CEFR_LEVELS * 100
    idx = 0
    
    while len(rows) < READING_PASSAGE_TARGET:
        topic = READING_TOPICS[idx % len(READING_TOPICS)]
        template = READING_PASSAGE_TEMPLATES[idx % len(READING_PASSAGE_TEMPLATES)]
        passage = template.format(topic=topic)
        level = levels_cycle[idx % len(levels_cycle)]
        idx += 1
        
        # Vary passages slightly
        variations = [
            f"Furthermore, experts suggest that further investment in {topic} is needed.",
            f"The government has announced new initiatives to support {topic} development.",
            f"International collaboration in {topic} has led to significant breakthroughs.",
            f"Public awareness of {topic} has increased dramatically in recent years.",
            f"The economic benefits of advancing {topic} are now widely recognised.",
        ]
        passage += ' ' + variations[idx % len(variations)]
        
        word_count = len(passage.split())
        ipa_val, _ = get_ipa(passage[:100], sentence_mode=True)  # IPA for first ~100 chars only
        diff = score_difficulty(passage, level, 'reading_passage')
        
        rows.append({
            'passage': passage, 'ipa': ipa_val, 'cefr_level': level,
            'topic': topic, 'word_count': word_count, 'difficulty': diff,
        })
    
    df = pd.DataFrame(rows[:READING_PASSAGE_TARGET])
    print(f"  Total reading passage entries: {len(df)}")
    return df


# ═══════════════════════════════ MINIMAL PAIR GENERATOR ══════════════════════

MINIMAL_PAIRS_SEED = [
    # Vowel contrasts
    ('ship', 'sheep', 'ʃɪp', 'ʃiːp'), ('sit', 'seat', 'sɪt', 'siːt'),
    ('bit', 'beat', 'bɪt', 'biːt'), ('hit', 'heat', 'hɪt', 'hiːt'),
    ('fill', 'feel', 'fɪl', 'fiːl'), ('live', 'leave', 'lɪv', 'liːv'),
    ('rich', 'reach', 'rɪtʃ', 'riːtʃ'), ('slip', 'sleep', 'slɪp', 'sliːp'),
    ('full', 'fool', 'fʊl', 'fuːl'), ('pull', 'pool', 'pʊl', 'puːl'),
    ('look', 'Luke', 'lʊk', 'luːk'), ('could', 'cooed', 'kʊd', 'kuːd'),
    ('hat', 'heart', 'hæt', 'hɑːt'), ('cap', 'carp', 'kæp', 'kɑːp'),
    ('cat', 'cart', 'kæt', 'kɑːt'), ('pack', 'park', 'pæk', 'pɑːk'),
    ('hat', 'hut', 'hæt', 'hʌt'), ('bat', 'but', 'bæt', 'bʌt'),
    ('ran', 'run', 'ræn', 'rʌn'), ('bad', 'bud', 'bæd', 'bʌd'),
    ('pet', 'pat', 'pet', 'pæt'), ('bed', 'bad', 'bed', 'bæd'),
    ('set', 'sat', 'set', 'sæt'), ('men', 'man', 'men', 'mæn'),
    ('pen', 'pan', 'pen', 'pæn'), ('ten', 'tan', 'ten', 'tæn'),
    ('cot', 'caught', 'kɒt', 'kɔːt'), ('pot', 'port', 'pɒt', 'pɔːt'),
    ('not', 'nought', 'nɒt', 'nɔːt'), ('fox', 'forks', 'fɒks', 'fɔːks'),
    ('cut', 'cot', 'kʌt', 'kɒt'), ('luck', 'lock', 'lʌk', 'lɒk'),
    ('cup', 'cop', 'kʌp', 'kɒp'), ('duck', 'dock', 'dʌk', 'dɒk'),
    # Consonant contrasts
    ('pin', 'bin', 'pɪn', 'bɪn'), ('pat', 'bat', 'pæt', 'bæt'),
    ('pack', 'back', 'pæk', 'bæk'), ('pill', 'bill', 'pɪl', 'bɪl'),
    ('tin', 'din', 'tɪn', 'dɪn'), ('town', 'down', 'taʊn', 'daʊn'),
    ('tie', 'die', 'taɪ', 'daɪ'), ('two', 'do', 'tuː', 'duː'),
    ('cap', 'gap', 'kæp', 'ɡæp'), ('coat', 'goat', 'kəʊt', 'ɡəʊt'),
    ('cold', 'gold', 'kəʊld', 'ɡəʊld'), ('curl', 'girl', 'kɜːl', 'ɡɜːl'),
    ('fan', 'van', 'fæn', 'væn'), ('few', 'view', 'fjuː', 'vjuː'),
    ('fine', 'vine', 'faɪn', 'vaɪn'), ('ferry', 'very', 'feri', 'veri'),
    ('thin', 'tin', 'θɪn', 'tɪn'), ('think', 'sink', 'θɪŋk', 'sɪŋk'),
    ('thick', 'tick', 'θɪk', 'tɪk'), ('three', 'tree', 'θriː', 'triː'),
    ('sin', 'shin', 'sɪn', 'ʃɪn'), ('sip', 'ship', 'sɪp', 'ʃɪp'),
    ('sue', 'shoe', 'suː', 'ʃuː'), ('sort', 'short', 'sɔːt', 'ʃɔːt'),
    ('cheap', 'jeep', 'tʃiːp', 'dʒiːp'), ('chin', 'gin', 'tʃɪn', 'dʒɪn'),
    ('choke', 'joke', 'tʃəʊk', 'dʒəʊk'), ('chain', 'Jane', 'tʃeɪn', 'dʒeɪn'),
    ('light', 'right', 'laɪt', 'raɪt'), ('lead', 'read', 'liːd', 'riːd'),
    ('long', 'wrong', 'lɒŋ', 'rɒŋ'), ('lip', 'rip', 'lɪp', 'rɪp'),
    ('wet', 'vet', 'wet', 'vet'), ('wine', 'vine', 'waɪn', 'vaɪn'),
    ('west', 'vest', 'west', 'vest'), ('weal', 'veal', 'wiːl', 'viːl'),
    # Voicing contrasts
    ('peace', 'peas', 'piːs', 'piːz'), ('rice', 'rise', 'raɪs', 'raɪz'),
    ('ice', 'eyes', 'aɪs', 'aɪz'), ('price', 'prize', 'praɪs', 'praɪz'),
    ('loose', 'lose', 'luːs', 'luːz'), ('use', 'use', 'juːs', 'juːz'),
    # Stress contrasts
    ('record', 'record', 'ˈrekɔːd', 'rɪˈkɔːd'),
    ('present', 'present', 'ˈprezənt', 'prɪˈzent'),
    ('contract', 'contract', 'ˈkɒntrækt', 'kənˈtrækt'),
    ('produce', 'produce', 'ˈprɒdjuːs', 'prəˈdjuːs'),
    ('object', 'object', 'ˈɒbdʒɪkt', 'əbˈdʒekt'),
    ('permit', 'permit', 'ˈpɜːmɪt', 'pəˈmɪt'),
    ('conduct', 'conduct', 'ˈkɒndʌkt', 'kənˈdʌkt'),
    ('project', 'project', 'ˈprɒdʒekt', 'prəˈdʒekt'),
    ('increase', 'increase', 'ˈɪŋkriːs', 'ɪnˈkriːs'),
    ('progress', 'progress', 'ˈprəʊɡres', 'prəˈɡres'),
]

def generate_minimal_pairs():
    """Generate 500+ minimal pair entries."""
    print("Generating Minimal Pairs Dataset...")
    rows = []
    levels_cycle = CEFR_LEVELS * 100
    seen = set()
    
    for idx, (w1, w2, ipa1, ipa2) in enumerate(MINIMAL_PAIRS_SEED):
        key = f"{w1}-{w2}"
        if key in seen:
            continue
        seen.add(key)
        level = levels_cycle[idx % len(levels_cycle)]
        diff = score_difficulty(f"{w1} {w2}", level, 'minimal_pair')
        rows.append({
            'word_1': w1, 'word_2': w2, 'ipa_1': ipa1, 'ipa_2': ipa2,
            'cefr_level': level, 'difficulty': diff,
        })
    
    # Expand with more pairs
    extra_pairs = [
        ('bat', 'bet'), ('cat', 'cut'), ('den', 'din'), ('fell', 'fill'),
        ('got', 'gut'), ('hall', 'hull'), ('jam', 'gem'), ('keen', 'king'),
        ('lot', 'let'), ('mat', 'met'), ('nap', 'nip'), ('pal', 'pull'),
        ('rap', 'rip'), ('sack', 'sock'), ('tag', 'tug'), ('wag', 'wig'),
        ('ban', 'bun'), ('fan', 'fun'), ('gap', 'gup'), ('map', 'mop'),
        ('sand', 'send'), ('band', 'bend'), ('lack', 'lick'), ('mass', 'mess'),
        ('pan', 'pin'), ('tap', 'tip'), ('back', 'beck'), ('bag', 'bug'),
        ('dam', 'dim'), ('ham', 'him'), ('lad', 'lid'), ('mad', 'mid'),
    ]
    
    idx = len(rows)
    for w1, w2 in extra_pairs:
        if len(rows) >= MINIMAL_PAIR_TARGET:
            break
        key = f"{w1}-{w2}"
        if key in seen:
            continue
        seen.add(key)
        ipa1, _ = get_ipa(w1)
        ipa2, _ = get_ipa(w2)
        level = levels_cycle[idx % len(levels_cycle)]
        diff = score_difficulty(f"{w1} {w2}", level, 'minimal_pair')
        rows.append({
            'word_1': w1, 'word_2': w2, 'ipa_1': ipa1, 'ipa_2': ipa2,
            'cefr_level': level, 'difficulty': diff,
        })
        idx += 1
    
    # Continue expanding if needed
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonant_starts = ['b', 'c', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'w']
    endings = ['t', 'n', 'p', 'd', 'g', 'l', 'k', 'sh', 'th']
    
    attempts = 0
    while len(rows) < MINIMAL_PAIR_TARGET and attempts < MINIMAL_PAIR_TARGET * 10:
        attempts += 1
        c = consonant_starts[(idx // 45) % len(consonant_starts)]
        v1 = vowels[(idx // 9) % len(vowels)]
        v2 = vowels[((idx // 9) + 1 + (idx % 4)) % len(vowels)]
        end = endings[idx % len(endings)]
        w1 = f"{c}{v1}{end}"
        w2 = f"{c}{v2}{end}"
        idx += 1
        key = f"{w1}-{w2}"
        if key in seen or w1 == w2:
            continue
        seen.add(key)
        ipa1, _ = get_ipa(w1)
        ipa2, _ = get_ipa(w2)
        level = levels_cycle[idx % len(levels_cycle)]
        diff = score_difficulty(f"{w1} {w2}", level, 'minimal_pair')
        rows.append({
            'word_1': w1, 'word_2': w2, 'ipa_1': ipa1, 'ipa_2': ipa2,
            'cefr_level': level, 'difficulty': diff,
        })
    
    df = pd.DataFrame(rows[:MINIMAL_PAIR_TARGET])
    print(f"  Total minimal pair entries: {len(df)}")
    return df


# ═══════════════════════════════ GRAMMAR GENERATOR ═══════════════════════════

GRAMMAR_EXAMPLES = {
    'Present Simple': [
        ("She works at a hospital.", "A1", "The present simple is used for habitual actions and general truths."),
        ("Water boils at 100 degrees Celsius.", "A1", "Facts and universal truths use the present simple tense."),
        ("They usually arrive at nine o'clock.", "A2", "Routine actions with frequency adverbs use the present simple."),
        ("The sun rises in the east.", "A1", "Scientific facts are expressed using the present simple."),
    ],
    'Present Continuous': [
        ("I am reading a book right now.", "A1", "The present continuous describes actions happening at this moment."),
        ("She is studying for her exam.", "A2", "Temporary actions in progress use the present continuous."),
        ("They are planning a conference for next month.", "B1", "Future arrangements often use the present continuous."),
        ("The company is expanding its operations.", "B2", "Ongoing processes use the present continuous."),
    ],
    'Present Perfect': [
        ("I have lived here for five years.", "A2", "The present perfect connects past experiences to the present."),
        ("She has already finished the report.", "B1", "Completed actions with present relevance use the present perfect."),
        ("They have been working on this project since January.", "B2", "Duration up to now uses the present perfect continuous."),
        ("The research has revealed significant findings.", "C1", "Academic writing often uses the present perfect for recent results."),
    ],
    'Past Simple': [
        ("She visited Paris last summer.", "A1", "The past simple describes completed actions in the past."),
        ("He graduated from university in 2020.", "A2", "Specific past dates use the past simple."),
        ("The team completed the project ahead of schedule.", "B1", "Past achievements use the past simple."),
        ("The government implemented new regulations.", "B2", "Historical events use the past simple."),
    ],
    'Past Continuous': [
        ("I was reading when the phone rang.", "A2", "The past continuous describes background actions interrupted by another event."),
        ("They were discussing the proposal when the manager arrived.", "B1", "Simultaneous past actions use the past continuous."),
        ("She was working on her thesis throughout the summer.", "B2", "Extended past actions use the past continuous."),
    ],
    'Past Perfect': [
        ("She had already left when I arrived.", "B1", "The past perfect shows an action completed before another past action."),
        ("They had finished the experiment before the deadline.", "B2", "Sequences of past events use the past perfect for the earlier action."),
        ("The committee had reviewed the proposal before making a decision.", "C1", "Formal narratives use the past perfect for sequencing."),
    ],
    'Future Forms': [
        ("I will call you tomorrow.", "A1", "Will + infinitive expresses future intentions and predictions."),
        ("She is going to study medicine.", "A2", "Going to expresses planned future actions."),
        ("The conference will be held next September.", "B1", "Scheduled future events use will + passive."),
        ("By 2030, renewable energy will have replaced fossil fuels.", "C1", "Future perfect expresses completion before a future point."),
    ],
    'Passive Voice': [
        ("The cake was made by my grandmother.", "A2", "The passive focuses on the object receiving the action."),
        ("The new bridge is being built by the council.", "B1", "Present continuous passive describes ongoing processes."),
        ("The results will be published next month.", "B2", "Future passive is common in academic and formal writing."),
        ("The phenomenon has been extensively studied.", "C1", "Present perfect passive emphasises research and findings."),
    ],
    'Conditionals': [
        ("If it rains, I will stay at home.", "A2", "First conditional expresses a real possibility and its likely result."),
        ("If I had more time, I would learn another language.", "B1", "Second conditional describes hypothetical present/future situations."),
        ("If she had studied harder, she would have passed.", "B2", "Third conditional discusses unreal past situations and their imagined results."),
        ("Had the government acted sooner, the crisis could have been averted.", "C1", "Inverted conditionals are used in formal academic writing."),
    ],
    'Relative Clauses': [
        ("The woman who lives next door is a doctor.", "A2", "Defining relative clauses identify the noun they modify."),
        ("The book, which was published last year, became a bestseller.", "B1", "Non-defining relative clauses add extra information."),
        ("The university at which she studied is world-renowned.", "B2", "Formal relative clauses use preposition + which."),
        ("The theory upon which the research is predicated requires further scrutiny.", "C1", "Academic relative clauses are highly formal."),
    ],
    'Reported Speech': [
        ("She said she was tired.", "A2", "Reported speech changes tenses one step back."),
        ("He told me that he had finished the project.", "B1", "Past perfect is used in reported speech for past actions."),
        ("The professor stated that the results confirmed the hypothesis.", "B2", "Academic reported speech uses formal reporting verbs."),
        ("The committee conceded that the methodology had been flawed.", "C1", "Formal reporting uses verbs like concede, assert, maintain."),
    ],
    'Modal Verbs': [
        ("You must wear a seatbelt.", "A1", "Must expresses obligation."),
        ("She can speak three languages.", "A2", "Can expresses ability."),
        ("You should consider all the options.", "B1", "Should expresses advice."),
        ("The results might indicate a correlation.", "B2", "Might expresses possibility in academic contexts."),
        ("One ought to consider the ethical implications.", "C1", "Ought to expresses moral obligation formally."),
    ],
    'Academic Grammar': [
        ("It is widely acknowledged that climate change poses a significant threat.", "B2", "Impersonal it + passive is common in academic writing."),
        ("The extent to which this affects outcomes remains unclear.", "C1", "Complex noun phrases are characteristic of academic English."),
        ("Notwithstanding the methodological limitations, the findings are robust.", "C1", "Concessive conjunctions are frequent in academic discourse."),
        ("The data, when subjected to rigorous analysis, yielded surprising results.", "C2", "Parenthetical clauses add nuance in formal academic writing."),
    ],
}

def generate_grammar():
    """Generate 2,000+ grammar example entries."""
    print("Generating Grammar Dataset...")
    rows = []
    seen = set()
    
    # Add seed examples
    for topic, examples in GRAMMAR_EXAMPLES.items():
        for sentence, level, explanation in examples:
            if sentence.lower() in seen:
                continue
            seen.add(sentence.lower())
            rows.append({
                'sentence': sentence, 'grammar_topic': topic,
                'cefr_level': level, 'explanation': explanation,
            })
    
    # Expand each topic with template variations
    subjects = ['The student', 'The team', 'The researchers', 'The company', 'The government',
                'The professor', 'The manager', 'She', 'He', 'They', 'We', 'The committee',
                'The organisation', 'The participants', 'The analyst']
    objects = ['the report', 'the project', 'the proposal', 'the data', 'the results',
               'the strategy', 'the findings', 'the analysis', 'the programme', 'the document',
               'the assessment', 'the methodology', 'the framework', 'the initiative']
    verbs_past = ['completed', 'analysed', 'submitted', 'reviewed', 'implemented',
                  'evaluated', 'published', 'presented', 'developed', 'established']
    verbs_present = ['completes', 'analyses', 'submits', 'reviews', 'implements',
                     'evaluates', 'publishes', 'presents', 'develops', 'establishes']
    
    idx = 0
    levels_cycle = CEFR_LEVELS * 500
    
    while len(rows) < GRAMMAR_TARGET:
        topic = GRAMMAR_TOPICS[idx % len(GRAMMAR_TOPICS)]
        subj = subjects[idx % len(subjects)]
        obj = objects[idx % len(objects)]
        vp = verbs_past[idx % len(verbs_past)]
        vpres = verbs_present[idx % len(verbs_present)]
        level = levels_cycle[idx % len(levels_cycle)]
        idx += 1
        
        if topic == 'Present Simple':
            sentence = f"{subj} {vpres} {obj} every week."
        elif topic == 'Present Continuous':
            sentence = f"{subj} is {vp.replace('ed', 'ing') if vp.endswith('ed') else vp + 'ing'} {obj} at the moment."
        elif topic == 'Present Perfect':
            sentence = f"{subj} has {vp} {obj} successfully."
        elif topic == 'Past Simple':
            sentence = f"{subj} {vp} {obj} last month."
        elif topic == 'Past Continuous':
            sentence = f"{subj} was working on {obj} when the announcement came."
        elif topic == 'Past Perfect':
            sentence = f"{subj} had already {vp} {obj} before the deadline."
        elif topic == 'Future Forms':
            sentence = f"{subj} will {vpres.rstrip('s')} {obj} next quarter."
        elif topic == 'Passive Voice':
            sentence = f"{obj.capitalize()} was {vp} by {subj.lower()}."
        elif topic == 'Conditionals':
            sentence = f"If {subj.lower()} {vp} {obj}, the outcomes would improve."
        elif topic == 'Relative Clauses':
            sentence = f"{subj}, who {vp} {obj}, received recognition."
        elif topic == 'Reported Speech':
            sentence = f"{subj} said that {obj} had been {vp}."
        elif topic == 'Modal Verbs':
            sentence = f"{subj} should {vpres.rstrip('s')} {obj} carefully."
        else:
            sentence = f"It is essential that {subj.lower()} {vpres.rstrip('s')} {obj}."
        
        if sentence.lower() in seen:
            continue
        seen.add(sentence.lower())
        
        explanation = f"This sentence demonstrates the use of {topic.lower()} in English."
        rows.append({
            'sentence': sentence, 'grammar_topic': topic,
            'cefr_level': level, 'explanation': explanation,
        })
    
    df = pd.DataFrame(rows[:GRAMMAR_TARGET])
    print(f"  Total grammar entries: {len(df)}")
    return df


# ═══════════════════════════════ DOMAIN GENERATORS ═══════════════════════════

DOMAIN_VOCAB = {
    'academic': [
        'thesis', 'dissertation', 'methodology', 'hypothesis', 'bibliography',
        'citation', 'abstract', 'peer review', 'empirical', 'qualitative',
        'quantitative', 'literature review', 'journal article', 'conference paper',
        'research proposal', 'academic integrity', 'plagiarism', 'footnote',
        'appendix', 'synopsis', 'annotation', 'corroboration', 'replication',
        'longitudinal study', 'cross-sectional study', 'meta-analysis',
        'systematic review', 'case study', 'field research', 'laboratory experiment',
        'control group', 'variable', 'correlation', 'causation', 'significance',
        'standard deviation', 'mean', 'median', 'probability', 'regression',
        'triangulation', 'grounded theory', 'phenomenology', 'ethnography',
        'discourse analysis', 'content analysis', 'narrative inquiry',
        'action research', 'participatory research', 'mixed methods',
    ],
    'business': [
        'accounting', 'audit', 'balance sheet', 'bankruptcy', 'bond',
        'budget', 'capital', 'cash flow', 'commodity', 'competitive advantage',
        'corporate governance', 'cost-benefit analysis', 'depreciation', 'dividend',
        'due diligence', 'earnings', 'equity', 'fiscal year', 'franchise',
        'gross profit', 'hedge fund', 'incentive', 'joint venture', 'leverage',
        'liability', 'liquidity', 'market share', 'merger', 'net profit',
        'outsourcing', 'overhead', 'portfolio', 'procurement', 'quarterly report',
        'receivables', 'revenue', 'shareholder', 'stakeholder', 'subsidiary',
        'supply chain', 'turnover', 'valuation', 'venture capital', 'workforce',
        'branding', 'consumer behaviour', 'demographics', 'elasticity',
        'entrepreneurship', 'globalisation', 'innovation',
    ],
    'technology': [
        'algorithm', 'API', 'artificial intelligence', 'backend', 'bandwidth',
        'blockchain', 'boolean', 'cache', 'cloud computing', 'compiler',
        'containerisation', 'cryptography', 'cybersecurity', 'data mining',
        'database', 'debugging', 'deep learning', 'deployment', 'DevOps',
        'encryption', 'endpoint', 'firewall', 'framework', 'frontend',
        'full stack', 'function', 'Git', 'GPU', 'HTML',
        'infrastructure', 'Internet of Things', 'iteration', 'JavaScript',
        'kernel', 'latency', 'machine learning', 'microservice', 'middleware',
        'neural network', 'open source', 'operating system', 'optimisation',
        'parallel processing', 'programming', 'protocol', 'Python',
        'quantum computing', 'refactoring', 'repository', 'scalability',
    ],
    'healthcare': [
        'anaesthesia', 'antibiotic', 'biopsy', 'cardiology', 'chemotherapy',
        'chronic disease', 'clinical trial', 'diagnosis', 'dosage', 'efficacy',
        'epidemiology', 'generic drug', 'haematology', 'immunisation',
        'intensive care', 'jaundice', 'kidney', 'laparoscopy', 'malignant',
        'neurological', 'oncology', 'outpatient', 'paediatrics', 'pathology',
        'pharmaceutical', 'physiotherapy', 'prognosis', 'prescription',
        'quarantine', 'radiology', 'rehabilitation', 'screening', 'surgery',
        'symptom', 'therapeutics', 'ultrasound', 'vaccination', 'ventilator',
        'ward', 'X-ray', 'adverse reaction', 'blood pressure', 'CT scan',
        'dehydration', 'electrocardiogram', 'fertility', 'gynaecology',
        'hormone', 'inflammation', 'liver function',
    ],
}

def _generate_domain_entries(domain, target):
    """Generate domain-specific entries."""
    vocab = DOMAIN_VOCAB.get(domain, [])
    rows = []
    levels_cycle = ['B1', 'B2', 'C1', 'C2'] * 1000
    idx = 0
    seen = set()
    
    # Base vocabulary entries
    for word in vocab:
        if word.lower() in seen:
            continue
        seen.add(word.lower())
        level = levels_cycle[idx % len(levels_cycle)]
        ipa_val, _ = get_ipa(word, sentence_mode=(' ' in word))
        cat = domain.capitalize()
        example = f"Understanding {word} is essential in the field of {domain}."
        diff = score_difficulty(word, level, 'word')
        rows.append({
            'vocabulary': word, 'phrase': f'{domain} {word}',
            'example_sentence': example, 'cefr_level': level, 'category': cat,
            'ipa': ipa_val, 'difficulty': diff,
        })
        idx += 1
    
    # Expand with phrase and sentence variations
    actions = ['analyse', 'evaluate', 'implement', 'develop', 'establish',
               'investigate', 'monitor', 'assess', 'review', 'optimise']
    contexts = ['in practice', 'in theory', 'in the workplace', 'in this context',
                'across industries', 'within organisations', 'for stakeholders',
                'in clinical settings', 'in academic research', 'for professionals']
    
    while len(rows) < target:
        base_word = vocab[(idx // 100) % len(vocab)]
        action = actions[(idx // 10) % len(actions)]
        context = contexts[idx % len(contexts)]
        level = levels_cycle[idx % len(levels_cycle)]
        idx += 1
        
        phrase = f"{action} {base_word} {context}"
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        
        example = f"It is important to {action} {base_word} {context}."
        ipa_val, _ = get_ipa(phrase, sentence_mode=True)
        diff = score_difficulty(phrase, level, 'phrase')
        
        rows.append({
            'vocabulary': base_word, 'phrase': phrase,
            'example_sentence': example, 'cefr_level': level,
            'category': domain.capitalize(),
            'ipa': ipa_val, 'difficulty': diff,
        })
    
    return pd.DataFrame(rows[:target])


def generate_domain_datasets():
    """Generate all four domain datasets."""
    print("Generating Domain Datasets...")
    
    academic_df = _generate_domain_entries('academic', ACADEMIC_TARGET)
    print(f"  Academic entries: {len(academic_df)}")
    
    business_df = _generate_domain_entries('business', BUSINESS_TARGET)
    print(f"  Business entries: {len(business_df)}")
    
    technology_df = _generate_domain_entries('technology', TECHNOLOGY_TARGET)
    print(f"  Technology entries: {len(technology_df)}")
    
    healthcare_df = _generate_domain_entries('healthcare', HEALTHCARE_TARGET)
    print(f"  Healthcare entries: {len(healthcare_df)}")
    
    return academic_df, business_df, technology_df, healthcare_df


# ═══════════════════════════════ AUDIO METADATA GENERATOR ════════════════════

def generate_audio_metadata(unified_df):
    """Generate audio metadata for all corpus entries."""
    print("Generating Audio Metadata...")
    random.seed(42)
    
    rows = []
    speaker_count = 200
    speakers = [f"SPK_{i:04d}" for i in range(1, speaker_count + 1)]
    speaker_genders = {s: random.choice(GENDERS) for s in speakers}
    
    for _, row in unified_df.iterrows():
        speaker = random.choice(speakers)
        rows.append({
            'id': row['id'],
            'text': row['text'],
            'ipa': row['ipa'],
            'cefr_level': row['cefr_level'],
            'category': row['category'],
            'speaker_id': speaker,
            'gender': speaker_genders[speaker],
            'accent': random.choice(ACCENTS),
            'speed': random.choice(SPEEDS),
            'emotion': random.choice(EMOTIONS),
        })
    
    df = pd.DataFrame(rows)
    print(f"  Total audio metadata entries: {len(df)}")
    return df
