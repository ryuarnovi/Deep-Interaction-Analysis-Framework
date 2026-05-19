import pdfplumber
import pandas as pd
import glob
import os
import re

def parse_oxford_pdfs(pdf_paths):
    all_words = []
    
    # We will track the current CEFR level. 
    # Usually it's A1, A2, B1, B2, C1, C2 on its own line or with large font.
    # To handle columns properly, we should group by columns or just read left-to-right but 
    # understand that the CEFR level changes.
    
    # Actually, a simpler way is to regex match every word with a POS tag.
    # The CEFR level is often printed as a heading. Let's see if we can find it.
    
    for path in pdf_paths:
        print(f"Processing {path}...")
        current_level = 'Unknown'
        
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # Extract words with coordinates to sort them column by column
                    words = page.extract_words(keep_blank_chars=True)
                    if not words: continue
                    
                    # Sort words primarily by x0 (column), then top (row)
                    # We can divide the page into 3 columns roughly
                    # Page width is usually ~595.
                    col1 = []
                    col2 = []
                    col3 = []
                    for w in words:
                        x = w['x0']
                        if x < 200: col1.append(w)
                        elif x < 400: col2.append(w)
                        else: col3.append(w)
                    
                    # Sort each column vertically
                    col1 = sorted(col1, key=lambda w: w['top'])
                    col2 = sorted(col2, key=lambda w: w['top'])
                    col3 = sorted(col3, key=lambda w: w['top'])
                    
                    # Combine columns in order
                    sorted_words = col1 + col2 + col3
                    
                    # Now extract text
                    text_ordered = " ".join([w['text'] for w in sorted_words])
                    
                    # The text is now linearly ordered. We can find the level markers (A1, A2, etc)
                    # and the words.
                    # A word entry looks like: "abandon v." or "ability n."
                    # We can iterate through tokens
                    tokens = text_ordered.split()
                    
                    pos_tags = {'n.', 'v.', 'adj.', 'adv.', 'prep.', 'conj.', 'pron.', 'det.', 'number', 'auxiliary', 'modal'}
                    
                    i = 0
                    while i < len(tokens):
                        token = tokens[i]
                        
                        # Check if token is a CEFR level
                        if token in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
                            current_level = token
                            i += 1
                            continue
                            
                        # Check if next token is a POS tag
                        if i + 1 < len(tokens) and (tokens[i+1] in pos_tags or tokens[i+1].replace(',', '') in pos_tags):
                            word = re.sub(r'[^a-zA-Z\-]', '', token).lower()
                            if len(word) > 0 and current_level != 'Unknown':
                                all_words.append({'word': word, 'cefr_level': current_level})
                            i += 2
                            continue
                            
                        i += 1
                        
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    df = pd.DataFrame(all_words)
    if len(df) > 0:
        # Keep the highest CEFR level if duplicates (or just drop duplicates)
        df = df.drop_duplicates(subset=['word'])
    
    return df

def main():
    pdf_files = glob.glob('models/*.pdf')
    df = parse_oxford_pdfs(pdf_files)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/oxford_vocabulary.csv'
    df.to_csv(out_path, index=False)
    
    print(f"Successfully extracted {len(df)} unique words to {out_path}.")
    if len(df) > 0:
        print(df.head(10))

if __name__ == "__main__":
    main()
