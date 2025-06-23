import re
import torch

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)  # Remove space before punctuation
        return text.strip()


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i: s for s, i in vocab.items() }

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [ item.strip() for item in preprocessed if item.strip() ]
        preprocessed = [ item if item in self.str_to_int else "<|unk|>" for item in preprocessed ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
class TokenizerV1:

    def encode(self, tokenizer, text):
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
        return encoded_tensor
    
    def decode(self, tokenizer, token_ids):
        flat = token_ids.squeeze(0).tolist()  # Remove batch dimension
        decoded = tokenizer.decode(flat)
        return decoded