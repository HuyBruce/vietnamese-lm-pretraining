"""Deterministic, bounded-memory text checks and hashed n-gram primitives."""
from collections import Counter
import hashlib
import re
import unicodedata
import numpy as np
from numba import njit

EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+', re.UNICODE)
PHONE = re.compile(r'(?<![\w\d])(?:\+84|0084|0)(?:[ .-]?[35789])(?:[ .-]?\d){8}(?![\w\d])')
IDENTIFIER = re.compile(r'(?i)(\b(?:CCCD|CMND|căn cước(?: công dân)?|chứng minh nhân dân|số tài khoản|STK)\s*[:#-]?\s*)(\d(?:[ .-]?\d){8,18})(?!\d)')
URL = re.compile(r'https?://\S+|www\.\S+', re.I)
WORDS = re.compile(r'\w+', re.UNICODE)

def dedup_text(text):
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())

def mask_pii(text):
    counts = {}
    text, counts['email'] = EMAIL.subn('<EMAIL>', text)
    # Contextual identifiers first: do not misclassify a bank account as a phone.
    text, counts['identifier'] = IDENTIFIER.subn(lambda m:m[1]+'<ID>', text)
    text, counts['phone'] = PHONE.subn('<PHONE>', text)
    return text, counts

CHAR_FLAGS=np.fromiter((int(chr(i).isalpha()) | (int(chr(i).isdigit())<<1) | (int(chr(i).isspace())<<2)
                       for i in range(0x110000)),dtype=np.uint8,count=0x110000)

@njit(cache=True)
def count_codepoints(codepoints, table):
    alpha=0; digits=0; spaces=0
    for c in codepoints:
        flags=table[c]
        alpha+=flags&1; digits+=(flags>>1)&1; spaces+=(flags>>2)&1
    return alpha,digits,spaces

def character_counts(text):
    return count_codepoints(np.frombuffer(text.encode('utf-32-le'),dtype='<u4'),CHAR_FLAGS)

def metrics(text):
    size = max(1,len(text))
    tokens = text.split()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    alpha,digits,spaces = character_counts(text)
    duplicate = sum(len(line)*(count-1) for line,count in Counter(lines).items())/size
    return dict(clean_char_count=len(text), whitespace_token_count=len(tokens),
                paragraph_count=len(lines), alphabetic_ratio=alpha/size,
                digit_ratio=digits/size, special_char_ratio=(len(text)-alpha-digits-spaces)/size,
                url_ratio=sum(len(m[0]) for m in URL.finditer(text))/size,
                duplicate_line_ratio=duplicate, duplicate_paragraph_ratio=duplicate,
                unique_token_ratio=len(set(tokens))/max(1,len(tokens)))

def reject_reasons(text, features, language, score, flags):
    reasons=[]
    if features['clean_char_count'] < 100 or features['whitespace_token_count'] < 20:
        reasons.append('too_short')
    if features['alphabetic_ratio'] < .4: reasons.append('low_alphabetic_ratio')
    if features['special_char_ratio'] > .3: reasons.append('excess_special_characters')
    if features['url_ratio'] > .3: reasons.append('excess_urls')
    if features['duplicate_line_ratio'] > .6: reasons.append('excess_repetition')
    if features['unique_token_ratio'] < .1: reasons.append('low_token_diversity')
    if language != 'vi': reasons.append('non_vietnamese')
    elif score < .5: reasons.append('uncertain_language')
    if '\ufffd' in text: reasons.append('encoding_replacement')
    if 'large_text_reduction_review' in flags or 'unclosed_html_preserved' in flags:
        reasons.append('normalization_review')
    return reasons

def word_bytes(text):
    return np.frombuffer(' '.join(WORDS.findall(unicodedata.normalize('NFC',text).lower())).encode('utf-8'), dtype=np.uint8)

@njit(cache=True)
def ngrams(data, n):
    # FNV-1a token hashes, then independent ordered n-gram hashes. Spaces delimit
    # normalized Unicode words; no Vietnamese word segmentation is introduced.
    tokens = np.empty(len(data)+1, np.uint64)
    count=0
    h=np.uint64(14695981039346656037)
    for b in data:
        if b==32:
            tokens[count]=h; count+=1; h=np.uint64(14695981039346656037)
        else:
            h=(h ^ np.uint64(b))*np.uint64(1099511628211)
    if len(data): tokens[count]=h; count+=1
    result=np.empty(max(0,count-n+1), np.uint64)
    for i in range(len(result)):
        h=np.uint64(14695981039346656037)
        for j in range(n): h=(h ^ tokens[i+j])*np.uint64(1099511628211)
        result[i]=h
    return np.unique(result)

RNG = np.random.default_rng(20260929)
PERM_A=RNG.integers(1,4294967291,64,dtype=np.uint64)
PERM_B=RNG.integers(0,4294967291,64,dtype=np.uint64)

@njit(cache=True)
def minhash(shingles):
    result=np.full(64,4294967295,np.uint32)
    for h in shingles:
        x=np.uint64(h & np.uint64(0xffffffff))
        for j in range(64):
            value=(PERM_A[j]*x+PERM_B[j]) % np.uint64(4294967291)
            if value<result[j]: result[j]=np.uint32(value)
    return result

@njit(cache=True)
def intersects(a,b):
    # b is a sorted benchmark index. O(len(a) log len(b)), small resident index.
    for x in a:
        j=np.searchsorted(b,x)
        if j<len(b) and b[j]==x: return True
    return False

@njit(cache=True)
def root(parents, i):
    while parents[i]!=i:
        parents[i]=parents[parents[i]]; i=parents[i]
    return i

@njit(cache=True)
def union_bucket(ids, parents, offsets, shingles, threshold):
    checked=0; merged=0
    for i in range(len(ids)):
        a=ids[i]; na=offsets[a+1]-offsets[a]
        for j in range(i+1,len(ids)):
            b=ids[j]
            ra=root(parents,a); rb=root(parents,b)
            if ra==rb: continue
            nb=offsets[b+1]-offsets[b]
            if min(na,nb)<threshold*max(na,nb): continue
            checked+=1
            ia=offsets[a]; ib=offsets[b]; matches=0
            while ia<offsets[a+1] and ib<offsets[b+1]:
                if shingles[ia]==shingles[ib]: matches+=1; ia+=1; ib+=1
                elif shingles[ia]<shingles[ib]: ia+=1
                else: ib+=1
            if matches>=threshold*(na+nb-matches):
                # Deterministic root; quality ranking chooses the survivor later.
                parents[max(ra,rb)]=min(ra,rb); merged+=1
    return checked, merged

@njit(cache=True)
def compress_roots(parents):
    for i in range(len(parents)): parents[i]=root(parents,i)
    return parents

def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
