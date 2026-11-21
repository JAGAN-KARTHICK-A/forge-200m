from pathlib import Path
from typing import List, Union

import sentencepiece as spm


class Tokenizer:
    def __init__(
        self,
        model_path: Union[str, Path] = "tokenizer/needle/tokenizer.model",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {self.model_path}")

        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

        self.vocab_size = self.sp.vocab_size()

        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        self.special_tokens = {}
        for token in (
            "<tool_call>",
            "</tool_call>",
            "<tools>",
            "</tools>",
        ):
            token_id = self.sp.piece_to_id(token)
            if token_id != self.unk_id or token == self.sp.id_to_piece(self.unk_id):
                if self.sp.id_to_piece(token_id) == token:
                    self.special_tokens[token] = token_id

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        ids = self.sp.encode(text, out_type=int)

        if add_bos and self.bos_id >= 0:
            ids.insert(0, self.bos_id)

        if add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode([int(token_id) for token_id in ids])

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp.id_to_piece(token_id)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_special_token_ids(self) -> dict:
        result = {
            "unk": self.unk_id,
            "bos": self.bos_id,
            "eos": self.eos_id,
            "pad": self.pad_id,
        }

        result.update(self.special_tokens)
        return result


"""
if __name__ == "__main__":
    tokenizer = Tokenizer()

    samples = [
        "Hello! How can I help you?",
        '{"name":"get_weather","arguments":{"location":"Chennai"}}',
        "<tool_call>",
    ]

    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Special token IDs:", tokenizer.get_special_token_ids())

    for sample in samples:
        ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(ids)

        print("\nText:   ", sample)
        print("IDs:    ", ids)
        print("Decoded:", decoded)

        assert decoded == sample, f"Round-trip failed for {sample!r}: {decoded!r}"
"""
