import os
import re
import sys
import pdfplumber
import pandas as pd

# Standard POS patterns to match parts of speech (allowing slash as lookahead boundary)
POS_PATTERN = r'\b(n\.|v\.|adj\.|adv\.|prep\.|conj\.|pron\.|det\.|exclam\.|number|auxiliary|modal|definite\s+article|indefinite\s+article|v|n|adj|adv)(?=\s|,|/|$)'

def is_pos_only_line(line_text):
    """
    Checks if a line consists ONLY of part of speech tags (useful for merging wrapped lines).
    """
    clean_text = line_text.strip().lower()
    if not clean_text:
        return False
        
    # Check if the clean_text matches the POS_PATTERN from index 0 to the end of the line
    match = re.match('^' + POS_PATTERN + '$', clean_text)
    if match:
        return True
        
    # Fallback checklist for combined tags like "n., adj." or "v., n."
    parts = re.split(r'[,/]', clean_text)
    std_pos = {
        'n', 'v', 'adj', 'adv', 'prep', 'conj', 'pron', 'det', 'exclam', 
        'number', 'auxiliary', 'modal', 'article', 'particle',
        'n.', 'v.', 'adj.', 'adv.', 'prep.', 'conj.', 'pron.', 'det.', 'exclam.'
    }
    if all(p.strip() in std_pos for p in parts if p.strip()):
        return True
        
    return False

def merge_wrapped_lines(lines):
    """
    Merges wrapped POS tags in a column into their preceding word entries.
    """
    merged = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # If it's a POS-only line and we have a previous line, append it to the previous line
        if is_pos_only_line(line_str) and merged:
            merged[-1] = merged[-1] + ' ' + line_str
        else:
            merged.append(line_str)
    return merged

def group_words_into_lines(words):
    """
    Groups pdfplumber words on the same line (within 1.5 point vertical tolerance)
    and sorts them horizontally by x0 coordinate.
    """
    lines = {}
    for w in words:
        matched = False
        for k in lines:
            if abs(k - w['top']) < 1.5:
                lines[k].append(w)
                matched = True
                break
        if not matched:
            lines[w['top']] = [w]
            
    # Sort lines by top coordinate and reconstruct line text
    sorted_lines_text = []
    for top in sorted(lines.keys()):
        line_words = sorted(lines[top], key=lambda w: w['x0'])
        line_text = ' '.join([w['text'] for w in line_words]).strip()
        sorted_lines_text.append(line_text)
    return sorted_lines_text

def parse_vocabulary_line(line_text, current_cefr_level):
    """
    Parses a single vocabulary line into word(s) and POS tags.
    Returns:
        - ('level', level_name) if it's a CEFR level marker
        - ('meta', None) if it's metadata (header, footer, page number, copyright)
        - ('entry', list_of_words, pos_tags) if it's a valid vocabulary entry
        - ('unparsed', line_text) if it could not be determined
    """
    line_text = line_text.strip()
    if not line_text:
        return None
        
    # Check for CEFR level markers
    if line_text in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        return ('level', line_text)
        
    # Check for metadata/headers/footers
    lower_text = line_text.lower()
    if ('oxford' in lower_text or 
        'cefr level' in lower_text or 
        'page' in lower_text or 
        'american english' in lower_text or
        re.match(r'^\d+\s*/\s*\d+$', line_text) or  # Page numbers like "4 / 13"
        'university press' in lower_text or
        'the oxford' in lower_text or
        'expanded core word list' in lower_text or
        'includes an additional' in lower_text or
        line_text.isdigit()):
        return ('meta', None)
        
    # Find POS tags in the line
    matches = list(re.finditer(POS_PATTERN, lower_text))
    # Filter out matches starting at index 0 (they cannot represent a POS tag since they leave the word empty)
    valid_matches = [m for m in matches if m.start() > 0]
    
    if valid_matches:
        first_match_start = valid_matches[0].start()
        word_part = line_text[:first_match_start].strip()
        pos_part = line_text[first_match_start:].strip()
        
        # Clean the word part:
        # 1. Remove trailing commas
        if word_part.endswith(','):
            word_part = word_part[:-1].strip()
            
        # 2. Strip homonym index digits like "can1" or "can2" at the end of word
        word_part = re.sub(r'\d+$', '', word_part)
        
        # 3. Remove parenthetical descriptions like (money) or (river)
        word_part = re.sub(r'\(.*?\)', '', word_part)
        
        # 4. Remove any non-alphanumeric/non-space/non-hyphen characters
        word_part = re.sub(r'[^a-zA-Z\s\-]', '', word_part)
        
        # 5. Normalize whitespace
        word_part = re.sub(r'\s+', ' ', word_part).strip()
        
        if not word_part:
            return ('unparsed', line_text)
            
        # Split compound words if they contain a comma but no parentheses (e.g. "a, an")
        if ',' in word_part and '(' not in word_part:
            sub_words = [w.strip() for w in word_part.split(',') if w.strip()]
        else:
            sub_words = [word_part]
            
        return ('entry', sub_words, pos_part)
    
    return ('unparsed', line_text)

def parse_pdf(pdf_path):
    print(f"Parsing PDF: {os.path.basename(pdf_path)}...")
    source_name = os.path.basename(pdf_path).replace('.pdf', '')
    
    all_entries = []
    unparsed_lines = []
    current_cefr_level = 'Unknown'
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            words = page.extract_words(keep_blank_chars=True)
            if not words:
                continue
                
            # Cluster words into columns based on x0 coordinate
            # Columns start at roughly x=42.5, 173.1, 303.6, 434.2
            col1 = [w for w in words if w['x0'] < 100]
            col2 = [w for w in words if 100 <= w['x0'] < 230]
            col3 = [w for w in words if 230 <= w['x0'] < 360]
            col4 = [w for w in words if w['x0'] >= 360]
            
            # Reconstruct lines for each column
            lines_col1 = merge_wrapped_lines(group_words_into_lines(col1))
            lines_col2 = merge_wrapped_lines(group_words_into_lines(col2))
            lines_col3 = merge_wrapped_lines(group_words_into_lines(col3))
            lines_col4 = merge_wrapped_lines(group_words_into_lines(col4))
            
            # Process column by column to maintain alphabetical / flow order
            all_page_lines = lines_col1 + lines_col2 + lines_col3 + lines_col4
            
            for line in all_page_lines:
                result = parse_vocabulary_line(line, current_cefr_level)
                if not result:
                    continue
                    
                res_type = result[0]
                if res_type == 'level':
                    current_cefr_level = result[1]
                elif res_type == 'meta':
                    continue
                elif res_type == 'entry':
                    _, sub_words, pos = result
                    for word in sub_words:
                        all_entries.append({
                            'word': word,
                            'cefr_level': current_cefr_level,
                            'pos': pos,
                            'source': source_name
                        })
                elif res_type == 'unparsed':
                    unparsed_lines.append((page_num, line))
                    
    print(f"  Parsed {len(all_entries)} raw entries.")
    if unparsed_lines:
        print(f"  Warning: {len(unparsed_lines)} lines could not be parsed:")
        for page_num, line in unparsed_lines[:10]:
            print(f"    Page {page_num}: {line}")
        if len(unparsed_lines) > 10:
            print(f"    ... and {len(unparsed_lines) - 10} more unparsed lines.")
            
    return all_entries, unparsed_lines

def clean_and_deduplicate(entries):
    print("Deduplicating vocabulary list...")
    df = pd.DataFrame(entries)
    
    if df.empty:
        return df
        
    # CEFR Level hierarchy for finding the lowest/earliest level
    level_hierarchy = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6, 'Unknown': 99}
    df['level_rank'] = df['cefr_level'].map(level_hierarchy)
    
    # Standardize word casing for grouping
    df['word_lower'] = df['word'].str.lower()
    
    # Custom aggregation logic per group (lowercased word)
    aggregated_rows = []
    for word_lower, group in df.groupby('word_lower'):
        # 1. Keep the row with the lowest CEFR Level
        best_row = group.loc[group['level_rank'].idxmin()]
        
        # 2. Merge all unique parts of speech
        all_pos = []
        for pos in group['pos']:
            if pd.isna(pos) or not pos:
                continue
            tags = re.split(r'[,/]', pos)
            for t in tags:
                t_clean = t.strip()
                if t_clean:
                    if not t_clean.endswith('.') and t_clean not in ['number', 'auxiliary', 'modal', 'article', 'particle']:
                        t_clean += '.'
                    if t_clean not in all_pos:
                        all_pos.append(t_clean)
        merged_pos = ', '.join(all_pos)
        
        # 3. Merge all unique source files
        unique_sources = group['source'].unique()
        merged_sources = ', '.join(unique_sources)
        
        aggregated_rows.append({
            'word': best_row['word'],
            'cefr_level': best_row['cefr_level'],
            'part_of_speech': merged_pos,
            'source_pdf': merged_sources
        })
        
    result_df = pd.DataFrame(aggregated_rows)
    
    # Sort alphabetically by word
    result_df = result_df.sort_values(by='word', key=lambda col: col.str.lower()).reset_index(drop=True)
    return result_df

def main():
    pdf_paths = [
        'models/American Oxford 3000 CEFR Levels.pdf',
        'models/American Oxford 5000 by CEFR Level.pdf'
    ]
    
    # Verify files exist
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Error: File '{path}' does not exist. Please check your current directory.")
            sys.exit(1)
            
    all_parsed_entries = []
    all_unparsed_lines = []
    
    for path in pdf_paths:
        entries, unparsed = parse_pdf(path)
        all_parsed_entries.extend(entries)
        all_unparsed_lines.extend(unparsed)
        
    # Clean and deduplicate
    final_df = clean_and_deduplicate(all_parsed_entries)
    
    # Write to CSV (lower columns for pipeline compatibility)
    os.makedirs('data', exist_ok=True)
    output_path = 'data/oxford_vocabulary.csv'
    
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully created CSV file at: {output_path}")
    print(f"Total Unique Words: {len(final_df)}")
    
    # Print stats by CEFR level
    if not final_df.empty:
        print("\nBreakdown by CEFR Level:")
        print(final_df['cefr_level'].value_counts().sort_index().to_string())
        
if __name__ == "__main__":
    main()
