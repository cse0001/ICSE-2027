# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for GPTNeoXJapanese."""

import collections
import json
import os
import re
import sys

import numpy as np

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file and emoji file into a dictionary."""
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()
    token = [[t.rstrip("\n")] if (t == "," or "," not in t) else t.rstrip("\n").split(",") for t in token]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx

    return vocab, raw_vocab, ids_to_tokens, emoji


class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] and is based on Japanese special Sub-Word-Encoding that is
    used in this repository (https://github.com/tanreinama/Japanese-BPEEncoder_V2). Check the repository for details.
    Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a
    combination of hiragana, katakana, and kanji, and variants such as "1" and "\u2460" are often used. In order to cope
    with these, this tokenizer has the following features
    - Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
    - BPEs are created for each Kanji, Hiragana, and Katakana character, and there are no BPEs that cross character
        types, such as Kanji + Hiragana or Hiragana + Katakana.
    - All-byte encoding that does not require <unk>.
    - Independent of UTF codes such as 2-byte and 3-byte characters
    - Conversion of heterographs to the same token_id
    - Emoji and Emoticon are grouped into 12 types as special tags.

    Example:

    ```python
    >>> from transformers import GPTNeoXJapaneseTokenizer

    >>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    >>> # You can confirm equivalent forms are normalized consistently.
    >>> tokenizer("Example text for tokenizer normalization.")["input_ids"]
    [30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]

    >>> # Decoding returns the normalized text.
    >>> tokenizer.decode(tokenizer("Example text for tokenizer normalization.")["input_ids"])
    'Example text for tokenizer normalization.'
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        emoji_file,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        do_clean_text=False,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_clean_text=do_clean_text,
            special_tokens_pattern="none",
            **kwargs,
        )

    @property
    def vocab_size(self):
        # self.vocab contains support for character fluctuation unique to Japanese, and has a large number of vocab
        return len(self.raw_vocab)

    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(",".join(token) + "\n")
                index += 1
        with open(emoji_file, "w", encoding="utf-8") as writer:
            json.dump(self.emoji, writer)
        return vocab_file, emoji_file


class SubWordJapaneseTokenizer:
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT License according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # same as swe
        self.ids_to_tokens = ids_to_tokens  # same as bpe
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-\u5e74])*(0?[1-9]|1[0-2])[/\-\u6708]((0?[1-9]|[12][0-9]|3[01])\u65e5?)*(\d{1,2}|:|\d{1,2}\u6642|\d{1,2}\u5206|\(\u65e5\)|\(\u6708\)|\(\u706b\)|\(\u6c34\)|\(\u6728\)|\(\u91d1\)|\(\u571f\)|\u3230|\u322a|\u322b|\u322c|\u322d|\u322e|\u322f)*"
        )
        self.content_repatter5 = re.compile(
            r"(\u660e\u6cbb|\u5927\u6b63|\u662d\u548c|\u5e73\u6210|\u4ee4\u548c|\u337e|\u337d|\u337c|\u337b|\u32ff)\d{1,2}\u5e74(0?[1-9]|1[0-2])\u6708(0?[1-9]|[12][0-9]|3[01])\u65e5(\d{1,2}|:|\d{1,2}\u6642|\d{1,2}\u5206|\(\u65e5\)|\(\u6708\)|\(\u706b\)|\(\u6c34\)|\(\u6728\)|\(\u91d1\)|\(\u571f\)|\u3230|\u322a|\u322b|\u322c|\u322d|\u322e|\u322f)*"
        )
        # The original version of this regex displays catastrophic backtracking behaviour. We avoid this using
        # possessive quantifiers in Py >= 3.11. In versions below this, we avoid the vulnerability using a slightly
        # different regex that should generally have the same behaviour in most non-pathological cases.
        if sys.version_info >= (3, 11):
            self.content_repatter6 = re.compile(
                r"(?:\d,\d{3}|[\d\u5104])*+"
                r"(?:\d,\d{3}|[\d\u4e07])*+"
                r"(?:\d,\d{3}|[\d\u5343])*+"
                r"(?:\u5343\u5186|\u4e07\u5186|\u5343\u4e07\u5186|\u5186|\u5343\u30c9\u30eb|\u4e07\u30c9\u30eb|\u5343\u4e07\u30c9\u30eb|\u30c9\u30eb|\u5343\u30e6\u30fc\u30ed|\u4e07\u30e6\u30fc\u30ed|\u5343\u4e07\u30e6\u30fc\u30ed|\u30e6\u30fc\u30ed)+"
                r"(?:\(\u7a0e\u8fbc\)|\(\u7a0e\u629c\)|\+tax)*"
            )
        else:
            self.content_repatter6 = re.compile(
                r"(?:\d,\d{3}|[\d\u5104\u4e07\u5343])*"
                r"(?:\u5343\u5186|\u4e07\u5186|\u5343\u4e07\u5186|\u5186|\u5343\u30c9\u30eb|\u4e07\u30c9\u30eb|\u5343\u4e07\u30c9\u30eb|\u30c9\u30eb|\u5343\u30e6\u30fc\u30ed|\u4e07\u30e6\u30fc\u30ed|\u5343\u4e07\u30e6\u30fc\u30ed|\u30e6\u30fc\u30ed)+"
                r"(?:\(\u7a0e\u8fbc\)|\(\u7a0e\u629c\)|\+tax)*"
            )
        keisen = "\u2500\u2501\u2502\u2503\u2504\u2505\u2506\u2507\u2508\u2509\u250a\u250b\u250c\u250d\u250e\u250f\u2510\u2511\u2512\u2513\u2514\u2515\u2516\u2517\u2518\u2519\u251a\u251b\u251c\u251d\u251e\u251f\u2520\u2521\u2522\u2523\u2524\u2525\u2526\u2527\u2528\u2529\u252a\u252b\u252c\u252d\u252e\u252f\u2530\u2531\u2532\u2533\u2534\u2535\u2536\u2537\u2538\u2539\u253a\u253b\u253c\u253d\u253e\u253f\u2540\u2541\u2542\u2543\u2544\u2545\u2546\u2547\u2548\u2549\u254a\u254b\u254c\u254d\u254e\u254f\u2550\u2551\u2552\u2553\u2554\u2555\u2556\u2557\u2558\u2559\u255a\u255b\u255c\u255d\u255e\u255f\u2560\u2561\u2562\u2563\u2564\u2565\u2566\u2567\u2568\u2569\u256a\u256b\u256c\u256d\u256e\u256f\u2570\u2571\u2572\u2573\u2574\u2575\u2576\u2577\u2578\u2579\u257a\u257b\u257c\u257d\u257e\u257f"
        blocks = "\u2580\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588\u2589\u258a\u258b\u258c\u258d\u258e\u258f\u2590\u2591\u2592\u2593\u2594\u2595\u2596\u2597\u2598\u2599\u259a\u259b\u259c\u259d\u259e\u259f"
        self.content_trans1 = str.maketrans(dict.fromkeys(keisen + blocks, "<BLOCK>"))

    def __len__(self):
        return len(self.ids_to_tokens)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)
        content = self.content_repatter2.sub("<EMAIL>", content)
        content = self.content_repatter3.sub("<TEL>", content)
        content = self.content_repatter4.sub("<DATE>", content)
        content = self.content_repatter5.sub("<DATE>", content)
        content = self.content_repatter6.sub("<PRICE>", content)
        content = content.translate(self.content_trans1)
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        return content

    def tokenize(self, text, clean=False):
        text = text.replace(" ", "<SP>")
        text = text.replace("\u3000", "<SP>")
        text = text.replace("\r\n", "<BR>")
        text = text.replace("\n", "<BR>")
        text = text.replace("\r", "<BR>")
        text = text.replace("\t", "<TAB>")
        text = text.replace("\u2014", "\u30fc")
        text = text.replace("\u2212", "\u30fc")
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False

        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False

        pos = 0
        result = []
        while pos < len(text):
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            candidates = []  # (token_id, token, pos)
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                # the smallest token_id is adopted
                _, wd, e = min(candidates, key=lambda x: x[0])
                result.append(wd)
                pos = e
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_simbol(wd):
                    result.append("<KIGOU>")
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        return result

    def convert_id_to_token(self, index, breakline="\n"):
        words = []
        byte_tokens = []
        word = self.ids_to_tokens[index][0]
        if word[:6] == "<|byte" and word[-2:] == "|>":
            byte_tokens.append(int(word[6:-2]))
        else:
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                byte_tokens = []
            if word[:7] == "<|emoji" and word[-2:] == "|>":
                words.append(self.emoji["emoji_inv"][word])
            elif word == "<SP>":
                words.append(" ")
            elif word == "<BR>":
                words.append(breakline)
            elif word == "<TAB>":
                words.append("\t")
            elif word == "<BLOCK>":
                words.append("\u2580")
            elif word == "<KIGOU>":
                words.append("\u01c0")
            elif word == "<U2000U2BFF>":
                words.append("\u2016")
            else:
                words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text


__all__ = ["GPTNeoXJapaneseTokenizer"]
