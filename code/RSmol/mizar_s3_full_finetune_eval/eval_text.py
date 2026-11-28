"""Existing evaluation prompt EOS suffix and right truncation/padding."""

class TextEncoder:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def tokenize(self, sentence, length):
        return self.tokenizer.encode_plus(
            text=sentence + " <|endoftext|>", add_special_tokens=True,
            max_length=length, padding="max_length", truncation=True, return_tensors="pt")
