# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"  # the llama sentencepiece tokenizer model
TOKENIZER_BIN = "tokenizer.bin"  # binary version of the tokenizer for inference in C


class Tokenizer:
    """Wrapper around the sentencepiece tokenizer."""

    def __init__(self):
        model_path = TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        # print(f"Loaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """Encode a string into a list of token IDs.

        Parameters
        ----------
        s
            String to encode.
        bos
            Beginning of sentence token.
        eos
            End of sentence token.

        Returns
        -------
        List[int]
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Decode a list of token IDs into a string.

        Parameters
        ----------
        t
            List of token IDs.

        Returns
        -------
        str
        """
        return self.sp_model.decode(t)

    def export(self):
        """Export the tokenizer to a binary file for inference in C."""

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = "\n<s>\n"
            elif i == self.eos_id:
                t = "\n</s>\n"
            elif len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                t = chr(int(t[3:5], 16))  # e.g. make '<0x01>' into '\x01'
            t = t.replace("▁", " ")  # sentencepiece uses this character as whitespace
            b = t.encode("utf-8")  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        with open(TOKENIZER_BIN, "wb") as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


if __name__ == "__main__":
    t = Tokenizer()
    t.export()
