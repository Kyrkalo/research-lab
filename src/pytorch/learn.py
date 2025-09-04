'''
Chapter 2: Working with Text Data
'''

import torch
import tiktoken
from importlib.metadata import version
import os
import urllib.request
import re

print(torch.__version__)
print(tiktoken.__version__)


if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])

'''
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no 
The goal is to tokenize and embed this text for an LLM
Let's develop a simple tokenizer based on some simple sample text that we can then later apply to the text above
The following regular expression will split on whitespaces
'''


text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
# print(len(result))
# print(result)

# We don't only want to split on whitespaces but also commas and periods, so let's modify the regular expression to do that as well
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]


#This looks pretty good, but let's also handle other types of punctuation, such as periods, question marks, and so on
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


print(len(preprocessed))

# Converting tokens into token IDs

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)

# print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""

from nlp.tokenizers import SimpleTokenizerV1, SimpleTokenizerV2
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))


#2.4 Adding special context tokens

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])


vocab = {token:integer for integer,token in enumerate(all_tokens)}
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

# print(text)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))


# 2.5 BytePair encoding
# 2.6 Data sampling with a sliding window
from datasets.GPTDatasetV1 import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)

first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)


print('----------------------------')
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


#2.7 Creating token embeddings

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))


# 47
print('The BytePair encoder has a vocabulary size of 50,257:')
print('Suppose we want to encode the input tokens into a 256-dimensional vector representation:')
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(token_embedding_layer.weight.shape)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# uncomment & execute the following line to see how the embedding layer weights look like
# print(pos_embedding_layer.weight)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
# print(pos_embeddings)