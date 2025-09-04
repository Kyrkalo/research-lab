
import tiktoken

from nlp.tokenizers import TokenizerV1

def test_tokenizer_v1():
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, world! This is a test."
    tokenizer_v1 = TokenizerV1()
    encoded = tokenizer_v1.encode(tokenizer, start_context)
    decoded = tokenizer_v1.decode(tokenizer, encoded)
    print(f"Encoded: {decoded}")

test_tokenizer_v1()