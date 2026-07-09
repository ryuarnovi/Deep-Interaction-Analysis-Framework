#!/usr/bin/env python3
"""
Oxford CEFR Master Corpus Builder
==================================
Master orchestrator that generates all datasets, assigns unique IDs,
merges into unified CSV, and produces corpus statistics.

Usage:
    python AI/corpus/build_corpus.py
"""

import os
import sys
import pandas as pd
import time

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from AI.corpus.vocabulary_generator import generate_vocabulary
from AI.corpus.generators import (
    generate_phrases,
    generate_sentences,
    generate_dialogues,
    generate_speaking_prompts,
    generate_reading_passages,
    generate_minimal_pairs,
    generate_grammar,
    generate_domain_datasets,
    generate_audio_metadata,
)


def assign_ids(df, prefix, id_col='id'):
    """Assign sequential unique IDs with a prefix."""
    df = df.copy()
    df[id_col] = [f"{prefix}_{i:05d}" for i in range(1, len(df) + 1)]
    return df


def unify_dataset(name, df, type_label, text_col, ipa_col='ipa', cat_col='category', diff_col='difficulty'):
    """Standardise a dataset into the unified schema."""
    unified = pd.DataFrame()
    unified['id'] = df['id']
    unified['text'] = df[text_col]
    unified['ipa'] = df.get(ipa_col, '')
    unified['cefr_level'] = df['cefr_level']
    unified['category'] = df.get(cat_col, 'General')
    unified['type'] = type_label
    unified['difficulty'] = df.get(diff_col, 50)
    return unified


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("  OXFORD CEFR MASTER CORPUS BUILDER")
    print("  World-class English CEFR corpus for Speech AI training")
    print("=" * 70)
    print()
    
    output_dir = os.path.join(project_root, 'data', 'corpus')
    os.makedirs(output_dir, exist_ok=True)
    
    # ──────────────── Generate all datasets ────────────────
    
    # 1. Vocabulary (14,000+ entries)
    vocab_df = generate_vocabulary(
        existing_vocab_path=os.path.join(project_root, 'data', 'oxford_vocabulary.csv')
    )
    vocab_df = assign_ids(vocab_df, 'VOCAB')
    vocab_df.to_csv(os.path.join(output_dir, 'vocabulary.csv'), index=False)
    print()
    
    # 2. Phrases (5,000+ entries)
    phrase_df = generate_phrases()
    phrase_df = assign_ids(phrase_df, 'PHRASE')
    phrase_df.to_csv(os.path.join(output_dir, 'phrases.csv'), index=False)
    print()
    
    # 3. Sentences (5,000+ entries)
    sentence_df = generate_sentences()
    sentence_df = assign_ids(sentence_df, 'SENT')
    sentence_df.to_csv(os.path.join(output_dir, 'sentences.csv'), index=False)
    print()
    
    # 4. Dialogues (2,000+ entries)
    dialogue_df = generate_dialogues()
    dialogue_df = assign_ids(dialogue_df, 'DIAL')
    dialogue_df.to_csv(os.path.join(output_dir, 'dialogues.csv'), index=False)
    print()
    
    # 5. Speaking Prompts (1,200+ entries)
    prompt_df = generate_speaking_prompts()
    prompt_df = assign_ids(prompt_df, 'PROMPT')
    prompt_df.to_csv(os.path.join(output_dir, 'speaking_prompts.csv'), index=False)
    print()
    
    # 6. Reading Passages (500+ entries)
    passage_df = generate_reading_passages()
    passage_df = assign_ids(passage_df, 'READ')
    passage_df.to_csv(os.path.join(output_dir, 'reading_passages.csv'), index=False)
    print()
    
    # 7. Minimal Pairs (500+ entries)
    pair_df = generate_minimal_pairs()
    pair_df = assign_ids(pair_df, 'MINPAIR')
    pair_df.to_csv(os.path.join(output_dir, 'minimal_pairs.csv'), index=False)
    print()
    
    # 8. Grammar Examples (2,000+ entries)
    grammar_df = generate_grammar()
    grammar_df = assign_ids(grammar_df, 'GRAM')
    grammar_df.to_csv(os.path.join(output_dir, 'grammar.csv'), index=False)
    print()
    
    # 9. Domain Datasets (3,000+ each)
    academic_df, business_df, technology_df, healthcare_df = generate_domain_datasets()
    academic_df = assign_ids(academic_df, 'ACAD')
    business_df = assign_ids(business_df, 'BIZ')
    technology_df = assign_ids(technology_df, 'TECH')
    healthcare_df = assign_ids(healthcare_df, 'HEALTH')
    academic_df.to_csv(os.path.join(output_dir, 'academic.csv'), index=False)
    business_df.to_csv(os.path.join(output_dir, 'business.csv'), index=False)
    technology_df.to_csv(os.path.join(output_dir, 'technology.csv'), index=False)
    healthcare_df.to_csv(os.path.join(output_dir, 'healthcare.csv'), index=False)
    print()
    
    # ──────────────── Merge into unified corpus ────────────────
    
    print("=" * 70)
    print("  BUILDING UNIFIED CORPUS")
    print("=" * 70)
    
    unified_parts = []
    
    # Vocabulary
    unified_parts.append(unify_dataset('vocabulary', vocab_df, 'word', 'word',
                                        diff_col='difficulty_score'))
    
    # Phrases
    unified_parts.append(unify_dataset('phrases', phrase_df, 'phrase', 'phrase'))
    
    # Sentences
    unified_parts.append(unify_dataset('sentences', sentence_df, 'sentence', 'sentence'))
    
    # Dialogues (combine speaker_a and speaker_b into text)
    dial_unified = pd.DataFrame()
    dial_unified['id'] = dialogue_df['id']
    dial_unified['text'] = dialogue_df['speaker_a'] + ' | ' + dialogue_df['speaker_b']
    dial_unified['ipa'] = dialogue_df['ipa_a']
    dial_unified['cefr_level'] = dialogue_df['cefr_level']
    dial_unified['category'] = dialogue_df['category']
    dial_unified['type'] = 'dialogue'
    dial_unified['difficulty'] = 50  # Default for dialogues
    unified_parts.append(dial_unified)
    
    # Speaking Prompts
    unified_parts.append(unify_dataset('prompts', prompt_df, 'speaking_prompt', 'prompt'))
    
    # Reading Passages
    unified_parts.append(unify_dataset('passages', passage_df, 'reading_passage', 'passage'))
    
    # Minimal Pairs (combine both words)
    pair_unified = pd.DataFrame()
    pair_unified['id'] = pair_df['id']
    pair_unified['text'] = pair_df['word_1'] + ' / ' + pair_df['word_2']
    pair_unified['ipa'] = pair_df['ipa_1'] + ' / ' + pair_df['ipa_2']
    pair_unified['cefr_level'] = pair_df['cefr_level']
    pair_unified['category'] = 'Pronunciation'
    pair_unified['type'] = 'minimal_pair'
    pair_unified['difficulty'] = pair_df['difficulty']
    unified_parts.append(pair_unified)
    
    # Grammar
    gram_unified = pd.DataFrame()
    gram_unified['id'] = grammar_df['id']
    gram_unified['text'] = grammar_df['sentence']
    gram_unified['ipa'] = ''
    gram_unified['cefr_level'] = grammar_df['cefr_level']
    gram_unified['category'] = grammar_df['grammar_topic']
    gram_unified['type'] = 'grammar'
    gram_unified['difficulty'] = 50
    unified_parts.append(gram_unified)
    
    # Domain datasets
    for domain_df, type_label in [(academic_df, 'academic'), (business_df, 'business'),
                                   (technology_df, 'technology'), (healthcare_df, 'healthcare')]:
        dom_unified = pd.DataFrame()
        dom_unified['id'] = domain_df['id']
        dom_unified['text'] = domain_df['vocabulary']
        dom_unified['ipa'] = domain_df['ipa']
        dom_unified['cefr_level'] = domain_df['cefr_level']
        dom_unified['category'] = domain_df['category']
        dom_unified['type'] = type_label
        dom_unified['difficulty'] = domain_df['difficulty']
        unified_parts.append(dom_unified)
    
    # Concatenate
    unified_df = pd.concat(unified_parts, ignore_index=True)
    
    # Reassign global unique IDs
    unified_df['id'] = [f"CEFR_{i:06d}" for i in range(1, len(unified_df) + 1)]
    
    # Save unified corpus
    unified_path = os.path.join(output_dir, 'cefr_master_corpus.csv')
    unified_df.to_csv(unified_path, index=False)
    print(f"Saved unified corpus to: {unified_path}")
    print(f"Total entries: {len(unified_df)}")
    
    # ──────────────── Generate Audio Metadata ────────────────
    
    audio_df = generate_audio_metadata(unified_df)
    audio_path = os.path.join(output_dir, 'audio_metadata.csv')
    audio_df.to_csv(audio_path, index=False)
    print(f"Saved audio metadata to: {audio_path}")
    print()
    
    # ──────────────── Print Statistics ────────────────
    
    elapsed = time.time() - start_time
    
    print("=" * 70)
    print("  CORPUS STATISTICS")
    print("=" * 70)
    print()
    print(f"{'Dataset':<25} {'Count':>8}")
    print("-" * 35)
    print(f"{'Vocabulary':<25} {len(vocab_df):>8,}")
    print(f"{'Phrases':<25} {len(phrase_df):>8,}")
    print(f"{'Sentences':<25} {len(sentence_df):>8,}")
    print(f"{'Dialogues':<25} {len(dialogue_df):>8,}")
    print(f"{'Speaking Prompts':<25} {len(prompt_df):>8,}")
    print(f"{'Reading Passages':<25} {len(passage_df):>8,}")
    print(f"{'Minimal Pairs':<25} {len(pair_df):>8,}")
    print(f"{'Grammar':<25} {len(grammar_df):>8,}")
    print(f"{'Academic':<25} {len(academic_df):>8,}")
    print(f"{'Business':<25} {len(business_df):>8,}")
    print(f"{'Technology':<25} {len(technology_df):>8,}")
    print(f"{'Healthcare':<25} {len(healthcare_df):>8,}")
    print("-" * 35)
    print(f"{'UNIFIED TOTAL':<25} {len(unified_df):>8,}")
    print(f"{'Audio Metadata':<25} {len(audio_df):>8,}")
    print()
    
    # CEFR distribution
    print("CEFR Level Distribution (Unified Corpus):")
    cefr_dist = unified_df['cefr_level'].value_counts().sort_index()
    for level, count in cefr_dist.items():
        pct = count / len(unified_df) * 100
        bar = '█' * int(pct)
        print(f"  {level}: {count:>7,} ({pct:>5.1f}%) {bar}")
    print()
    
    # Type distribution
    print("Entry Type Distribution:")
    type_dist = unified_df['type'].value_counts()
    for entry_type, count in type_dist.items():
        print(f"  {entry_type:<20} {count:>7,}")
    print()
    
    # Validation
    print("=" * 70)
    print("  VALIDATION")
    print("=" * 70)
    
    # Check for duplicate IDs
    dup_ids = unified_df['id'].duplicated().sum()
    print(f"  Duplicate IDs: {dup_ids} {'✓' if dup_ids == 0 else '✗ FAILED'}")
    
    # Check CEFR levels
    valid_cefr = set(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    invalid_cefr = set(unified_df['cefr_level'].unique()) - valid_cefr
    print(f"  Invalid CEFR levels: {len(invalid_cefr)} {'✓' if len(invalid_cefr) == 0 else '✗ ' + str(invalid_cefr)}")
    
    # Check all CEFR levels present
    missing_cefr = valid_cefr - set(unified_df['cefr_level'].unique())
    print(f"  Missing CEFR levels: {len(missing_cefr)} {'✓' if len(missing_cefr) == 0 else '✗ ' + str(missing_cefr)}")
    
    # Check empty text
    empty_text = unified_df['text'].isna().sum() + (unified_df['text'] == '').sum()
    print(f"  Empty text entries: {empty_text} {'✓' if empty_text == 0 else '✗ FAILED'}")
    
    # Check IPA
    non_empty_ipa = (unified_df['ipa'] != '').sum()
    ipa_coverage = non_empty_ipa / len(unified_df) * 100
    print(f"  IPA coverage: {non_empty_ipa:,} / {len(unified_df):,} ({ipa_coverage:.1f}%)")
    
    print()
    print(f"Build completed in {elapsed:.1f} seconds.")
    print("=" * 70)
    
    # Output file listing
    print()
    print("Generated files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:<35} {size_kb:>8.1f} KB")


if __name__ == '__main__':
    main()
