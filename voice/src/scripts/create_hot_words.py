import re


words = set()
with open('hotwords_text.txt') as hotwords_text:
    for line in hotwords_text:
        line_words = re.sub(r'[^a-z\s]', '', line.rstrip().lower()).split()
        words |= set(line_words)
for word in words:
    print(word)