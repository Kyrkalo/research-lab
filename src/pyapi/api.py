from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tiktoken

app = FastAPI()
tokenizer = tiktoken.get_encoding("gpt2")


class EncodeRequest(BaseModel):
    text: str


class DecodeRequest(BaseModel):
    tokens: List[int]


@app.post("/encode")
def encode_text(req: EncodeRequest):
    ids = tokenizer.encode(req.text)
    return {"tokens": ids}


@app.post("/decode")
def decode_tokens(req: DecodeRequest):
    text = tokenizer.decode(req.tokens)
    return {"text": text}
# uvicorn api:app --host 127.0.0.1 --port 8000