"""
IPA Transcription Engine for the Oxford CEFR Master Corpus.
Uses eng_to_ipa (CMU-based) as baseline with a curated British RP override dictionary.
"""

import re
try:
    import eng_to_ipa as ipa
except ImportError:
    ipa = None

# ─────────────── British RP Override Dictionary (~500 common words) ───────────
# Where British RP differs significantly from American English (CMU).
# Format: word -> British RP IPA

BRITISH_RP_OVERRIDES = {
    # Vowel shifts (BATH/TRAP, LOT/CLOTH, etc.)
    'dance': 'dɑːns', 'chance': 'tʃɑːns', 'advance': 'ədˈvɑːns',
    'bath': 'bɑːθ', 'path': 'pɑːθ', 'class': 'klɑːs', 'glass': 'ɡlɑːs',
    'grass': 'ɡrɑːs', 'pass': 'pɑːs', 'past': 'pɑːst', 'last': 'lɑːst',
    'fast': 'fɑːst', 'ask': 'ɑːsk', 'task': 'tɑːsk', 'mask': 'mɑːsk',
    'after': 'ˈɑːftə', 'half': 'hɑːf', 'laugh': 'lɑːf', 'staff': 'stɑːf',
    'example': 'ɪɡˈzɑːmpəl', 'answer': 'ˈɑːnsə', 'command': 'kəˈmɑːnd',
    'demand': 'dɪˈmɑːnd', 'plant': 'plɑːnt', 'grant': 'ɡrɑːnt',
    'can\'t': 'kɑːnt', 'rather': 'ˈrɑːðə', 'father': 'ˈfɑːðə',
    'castle': 'ˈkɑːsəl', 'basket': 'ˈbɑːskɪt', 'master': 'ˈmɑːstə',
    'disaster': 'dɪˈzɑːstə', 'advantage': 'ədˈvɑːntɪdʒ',
    # Non-rhotic (no final /r/)
    'car': 'kɑː', 'far': 'fɑː', 'star': 'stɑː', 'bar': 'bɑː',
    'heart': 'hɑːt', 'start': 'stɑːt', 'part': 'pɑːt', 'art': 'ɑːt',
    'water': 'ˈwɔːtə', 'letter': 'ˈletə', 'better': 'ˈbetə',
    'teacher': 'ˈtiːtʃə', 'computer': 'kəmˈpjuːtə', 'never': 'ˈnevə',
    'ever': 'ˈevə', 'over': 'ˈəʊvə', 'under': 'ˈʌndə',
    'mother': 'ˈmʌðə', 'brother': 'ˈbrʌðə', 'other': 'ˈʌðə',
    'number': 'ˈnʌmbə', 'remember': 'rɪˈmembə', 'together': 'təˈɡeðə',
    'order': 'ˈɔːdə', 'important': 'ɪmˈpɔːtənt', 'report': 'rɪˈpɔːt',
    'support': 'səˈpɔːt', 'information': 'ˌɪnfəˈmeɪʃən',
    'door': 'dɔː', 'floor': 'flɔː', 'more': 'mɔː', 'before': 'bɪˈfɔː',
    'four': 'fɔː', 'poor': 'pɔː', 'your': 'jɔː',
    'here': 'hɪə', 'near': 'nɪə', 'clear': 'klɪə', 'idea': 'aɪˈdɪə',
    'year': 'jɪə', 'appear': 'əˈpɪə', 'beer': 'bɪə',
    'air': 'eə', 'care': 'keə', 'where': 'weə', 'there': 'ðeə',
    'share': 'ʃeə', 'prepare': 'prɪˈpeə', 'aware': 'əˈweə',
    'sure': 'ʃɔː', 'tour': 'tʊə', 'pure': 'pjʊə',
    # LOT vowel (AmE /ɑː/ -> BrE /ɒ/)
    'hot': 'hɒt', 'not': 'nɒt', 'got': 'ɡɒt', 'lot': 'lɒt',
    'stop': 'stɒp', 'top': 'tɒp', 'shop': 'ʃɒp', 'job': 'dʒɒb',
    'body': 'ˈbɒdi', 'problem': 'ˈprɒbləm', 'product': 'ˈprɒdʌkt',
    'project': 'ˈprɒdʒekt', 'process': 'ˈprəʊses', 'possible': 'ˈpɒsɪbəl',
    'knowledge': 'ˈnɒlɪdʒ', 'college': 'ˈkɒlɪdʒ', 'policy': 'ˈpɒləsi',
    'popular': 'ˈpɒpjʊlə', 'politics': 'ˈpɒlətɪks', 'economy': 'ɪˈkɒnəmi',
    'holiday': 'ˈhɒlədeɪ', 'hospital': 'ˈhɒspɪtəl', 'offer': 'ˈɒfə',
    'office': 'ˈɒfɪs', 'officer': 'ˈɒfɪsə', 'operate': 'ˈɒpəreɪt',
    'option': 'ˈɒpʃən', 'obvious': 'ˈɒbviəs', 'opposite': 'ˈɒpəzɪt',
    'quality': 'ˈkwɒləti', 'quantity': 'ˈkwɒntəti',
    'technology': 'tekˈnɒlədʒi', 'philosophy': 'fɪˈlɒsəfi',
    'democracy': 'dɪˈmɒkrəsi', 'document': 'ˈdɒkjʊmənt',
    # Specific pronunciation differences
    'schedule': 'ˈʃedjuːl', 'leisure': 'ˈleʒə', 'lieutenant': 'lefˈtenənt',
    'garage': 'ˈɡærɑːʒ', 'privacy': 'ˈprɪvəsi', 'vitamin': 'ˈvɪtəmɪn',
    'tomato': 'təˈmɑːtəʊ', 'advertisement': 'ədˈvɜːtɪsmənt',
    'aluminium': 'ˌæljʊˈmɪniəm', 'mobile': 'ˈməʊbaɪl',
    'route': 'ruːt', 'either': 'ˈaɪðə', 'neither': 'ˈnaɪðə',
    'been': 'biːn', 'again': 'əˈɡen', 'herb': 'hɜːb',
    # GOAT vowel (AmE /oʊ/ -> BrE /əʊ/)
    'go': 'ɡəʊ', 'no': 'nəʊ', 'so': 'səʊ', 'know': 'nəʊ',
    'show': 'ʃəʊ', 'home': 'həʊm', 'phone': 'fəʊn', 'alone': 'əˈləʊn',
    'close': 'kləʊs', 'note': 'nəʊt', 'hope': 'həʊp', 'open': 'ˈəʊpən',
    'whole': 'həʊl', 'control': 'kənˈtrəʊl', 'role': 'rəʊl',
    'programme': 'ˈprəʊɡræm', 'focus': 'ˈfəʊkəs', 'approach': 'əˈprəʊtʃ',
    'social': 'ˈsəʊʃəl', 'local': 'ˈləʊkəl', 'total': 'ˈtəʊtəl',
    'global': 'ˈɡləʊbəl', 'growth': 'ɡrəʊθ', 'own': 'əʊn',
    # Common everyday words
    'the': 'ðə', 'a': 'ə', 'an': 'ən', 'is': 'ɪz', 'are': 'ɑː',
    'was': 'wɒz', 'were': 'wɜː', 'have': 'hæv', 'has': 'hæz',
    'do': 'duː', 'does': 'dʌz', 'did': 'dɪd', 'will': 'wɪl',
    'would': 'wʊd', 'could': 'kʊd', 'should': 'ʃʊd', 'must': 'mʌst',
    'can': 'kæn', 'may': 'meɪ', 'might': 'maɪt', 'shall': 'ʃæl',
    'I': 'aɪ', 'you': 'juː', 'he': 'hiː', 'she': 'ʃiː', 'it': 'ɪt',
    'we': 'wiː', 'they': 'ðeɪ', 'me': 'miː', 'him': 'hɪm',
    'her': 'hɜː', 'us': 'ʌs', 'them': 'ðem',
    'this': 'ðɪs', 'that': 'ðæt', 'these': 'ðiːz', 'those': 'ðəʊz',
    'my': 'maɪ', 'his': 'hɪz', 'its': 'ɪts', 'our': 'ˈaʊə',
    'their': 'ðeə', 'what': 'wɒt', 'which': 'wɪtʃ', 'who': 'huː',
    'how': 'haʊ', 'when': 'wen', 'why': 'waɪ',
    'good': 'ɡʊd', 'new': 'njuː', 'first': 'fɜːst', 'long': 'lɒŋ',
    'great': 'ɡreɪt', 'little': 'ˈlɪtəl', 'own': 'əʊn',
    'old': 'əʊld', 'right': 'raɪt', 'big': 'bɪɡ', 'high': 'haɪ',
    'small': 'smɔːl', 'large': 'lɑːdʒ', 'young': 'jʌŋ',
    'different': 'ˈdɪfrənt', 'important': 'ɪmˈpɔːtənt',
    'world': 'wɜːld', 'life': 'laɪf', 'hand': 'hænd', 'work': 'wɜːk',
    'place': 'pleɪs', 'case': 'keɪs', 'week': 'wiːk', 'company': 'ˈkʌmpəni',
    'system': 'ˈsɪstəm', 'program': 'ˈprəʊɡræm', 'question': 'ˈkwestʃən',
    'government': 'ˈɡʌvənmənt', 'country': 'ˈkʌntri', 'story': 'ˈstɔːri',
    'fact': 'fækt', 'month': 'mʌnθ', 'study': 'ˈstʌdi',
    'book': 'bʊk', 'eye': 'aɪ', 'word': 'wɜːd', 'money': 'ˈmʌni',
    'business': 'ˈbɪznɪs', 'issue': 'ˈɪʃuː', 'side': 'saɪd',
    'kind': 'kaɪnd', 'head': 'hed', 'house': 'haʊs', 'service': 'ˈsɜːvɪs',
    'friend': 'frend', 'power': 'ˈpaʊə', 'hour': 'ˈaʊə',
    'game': 'ɡeɪm', 'line': 'laɪn', 'end': 'end', 'member': 'ˈmembə',
    'law': 'lɔː', 'war': 'wɔː', 'car': 'kɑː', 'city': 'ˈsɪti',
    'community': 'kəˈmjuːnəti', 'name': 'neɪm',
    # Academic / technical
    'research': 'rɪˈsɜːtʃ', 'university': 'ˌjuːnɪˈvɜːsəti',
    'analysis': 'əˈnæləsɪs', 'data': 'ˈdeɪtə', 'theory': 'ˈθɪəri',
    'development': 'dɪˈveləpmənt', 'environment': 'ɪnˈvaɪrənmənt',
    'education': 'ˌedjʊˈkeɪʃən', 'experience': 'ɪkˈspɪəriəns',
    'international': 'ˌɪntəˈnæʃənəl', 'organisation': 'ˌɔːɡənaɪˈzeɪʃən',
    'performance': 'pəˈfɔːməns', 'management': 'ˈmænɪdʒmənt',
    'economic': 'ˌiːkəˈnɒmɪk', 'financial': 'faɪˈnænʃəl',
    'professional': 'prəˈfeʃənəl', 'strategy': 'ˈstrætədʒi',
    'assessment': 'əˈsesmənt', 'communication': 'kəˌmjuːnɪˈkeɪʃən',
    'investment': 'ɪnˈvestmənt', 'application': 'ˌæplɪˈkeɪʃən',
    'achievement': 'əˈtʃiːvmənt', 'artificial': 'ˌɑːtɪˈfɪʃəl',
    'intelligence': 'ɪnˈtelɪdʒəns', 'algorithm': 'ˈælɡərɪðəm',
    'infrastructure': 'ˈɪnfrəstrʌktʃə', 'pharmaceutical': 'ˌfɑːməˈsjuːtɪkəl',
    'pronunciation': 'prəˌnʌnsiˈeɪʃən', 'vocabulary': 'vəˈkæbjʊləri',
    'grammar': 'ˈɡræmə', 'fluency': 'ˈfluːənsi',
    'laboratory': 'ləˈbɒrətri', 'certificate': 'səˈtɪfɪkət',
    'mathematics': 'ˌmæθəˈmætɪks', 'literature': 'ˈlɪtrətʃə',
}


def get_ipa(text, sentence_mode=False):
    """
    Get IPA transcription for a word or sentence.
    Uses British RP overrides first, then falls back to eng_to_ipa.
    
    Returns: (ipa_string, source)
        source is 'curated_brp' or 'auto_cmu'
    """
    text_clean = text.strip()
    if not text_clean:
        return ('', 'none')

    if sentence_mode:
        return _get_sentence_ipa(text_clean)
    
    return _get_word_ipa(text_clean)


_word_ipa_cache = {}

def _get_word_ipa(word):
    """Get IPA for a single word."""
    word_lower = word.lower().strip()
    if word_lower in _word_ipa_cache:
        return _word_ipa_cache[word_lower]
    
    # Check British RP override first
    if word_lower in BRITISH_RP_OVERRIDES:
        res = (BRITISH_RP_OVERRIDES[word_lower], 'curated_brp')
        _word_ipa_cache[word_lower] = res
        return res
    
    # Fallback to eng_to_ipa
    if ipa:
        try:
            result = ipa.convert(word_lower)
            if result and '*' not in result:
                res = (result, 'auto_cmu')
                _word_ipa_cache[word_lower] = res
                return res
        except Exception:
            pass
    
    res = (word_lower, 'unknown')
    _word_ipa_cache[word_lower] = res
    return res



def _get_sentence_ipa(sentence):
    """Get IPA for a full sentence, word by word."""
    # Remove punctuation for IPA but preserve spacing
    words = re.findall(r"[a-zA-Z'-]+", sentence)
    ipa_parts = []
    sources = set()
    
    for w in words:
        w_ipa, source = _get_word_ipa(w)
        ipa_parts.append(w_ipa)
        sources.add(source)
    
    combined_ipa = ' '.join(ipa_parts)
    primary_source = 'curated_brp' if 'curated_brp' in sources else 'auto_cmu'
    
    return (combined_ipa, primary_source)
