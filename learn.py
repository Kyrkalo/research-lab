'''
Chapter 2: Working with Text Data
'''

import torch
import tiktoken
from importlib.metadata import version

print(torch.__version__)
print(tiktoken.__version__)

import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

'''
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
The goal is to tokenize and embed this text for an LLM
Let's develop a simple tokenizer based on some simple sample text that we can then later apply to the text above
The following regular expression will split on whitespaces
'''

import re

text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
# print(len(result))
# print(result)

# We don't only want to split on whitespaces but also commas and periods, so let's modify the regular expression to do that as well
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print(result)

#This looks pretty good, but let's also handle other types of punctuation, such as periods, question marks, and so on
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed)
print(len(preprocessed))

# Converting tokens into token IDs

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

from nlp.tokenizers import SimpleTokenizerV1
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))


#2.4 Adding special context tokens
