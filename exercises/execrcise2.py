from importlib.metadata import version
import torch
import tiktoken
print("torch version:", torch.__version__)
print("tiktoken version:", tiktoken.__version__)

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode("Akwirw ier")
print(integers)
for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")