"""Regenerate the compact Kimi K3 rank fixture from the pinned full tokenizer.

Run from the repository root with the downloaded tokenizer.json path as argv[1].
Source: https://huggingface.co/Xenova/Kimi-K3-tokenizer/resolve/3f11cbe14873d9b64356607af3bbef7608aadf5b/tokenizer.json
"""
import hashlib
import json
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("usage: generate_kimi_k3_rank_reference.py <tokenizer.json>")
source = Path(sys.argv[1])
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "b55c4532c501114da9a8891b77d244fa32eee2ace2bd51abb5dc4fb156f60eb9"
full = json.loads(raw)
vocab = full["model"]["vocab"]
split = json.loads(Path("internal/tokenizer/testdata/kimi-declared-split.json").read_text(encoding="utf-8"))

base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
byte_to_unicode = {value: chr(value) for value in base}
next_codepoint = 256
for value in range(256):
    if value not in byte_to_unicode:
        byte_to_unicode[value] = chr(next_codepoint)
        next_codepoint += 1


def encoded_piece(piece):
    return "".join(byte_to_unicode[value] for value in piece.encode("utf-8"))


def rank_bpe(piece):
    encoded = encoded_piece(piece)
    if encoded in vocab:
        return [vocab[encoded]]
    symbols = list(encoded)
    while len(symbols) > 1:
        candidate = min(
            ((vocab[symbols[i] + symbols[i + 1]], i)
             for i in range(len(symbols) - 1)
             if symbols[i] + symbols[i + 1] in vocab),
            default=None,
        )
        if candidate is None:
            break
        _, index = candidate
        symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]
    return [vocab[symbol] for symbol in symbols]


cases = [
    {"input": case["input"], "pieces": case["pieces"]}
    for case in split["cases"]
]
cases.extend([
    {"input": "newlines", "pieces": ["newlines"]},
    {"input": "HTTPResponse", "pieces": ["HTTPResponse"]},
])
for case in cases:
    assert "".join(case["pieces"]) == case["input"]
    case["ids"] = [item for piece in case["pieces"] for item in rank_bpe(piece)]

wire_pieces = [encoded_piece(piece) for case in cases for piece in case["pieces"]]
subset = {
    token: rank for token, rank in vocab.items()
    if token and any(token in piece for piece in wire_pieces)
}
fixture = {
    "source": "Xenova/Kimi-K3-tokenizer tokenizer.json",
    "source_commit": "3f11cbe14873d9b64356607af3bbef7608aadf5b",
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "split_source_sha256": split["source_sha256"],
    "pattern": split["pattern"],
    "vocab": subset,
    "added_tokens": [
        token for token in full["added_tokens"]
        if token["content"] in ("[BOS]", "<|end_of_msg|>", "<|open|>")
    ],
    "cases": cases,
}
out = Path("internal/hfbpe/testdata/kimi_k3_rank_reference.json")
out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"vocab {len(vocab)} -> {len(subset)} tokens; {len(cases)} cases; {out.stat().st_size} bytes")
for case in cases:
    print(json.dumps(case["input"], ensure_ascii=True), case["ids"])

literal_pieces = ["hello", " <|", "end", "_of", "_msg", "|>", " world"]
literal_text = "hello <|end_of_msg|> world"
assert "".join(literal_pieces) == literal_text
literal_ids = [item for piece in literal_pieces for item in rank_bpe(piece)]
assert literal_ids == [22931, 22652, 517, 5118, 14222, 91, 29, 2695]
literal_wire = [encoded_piece(piece) for piece in literal_pieces] + [encoded_piece("<|open|>")]
literal_fixture = {
    "source_commit": fixture["source_commit"],
    "source_sha256": fixture["source_sha256"],
    "published_reference": "https://huggingface.co/moonshotai/Kimi-K3/discussions/60",
    "literal_text": literal_text,
    "literal_pieces": literal_pieces,
    "literal_ids": literal_ids,
    "structural_ids": [22931, 220, 163586, 2695],
    "ordinary_added_text": "<|open|>",
    "ordinary_added_id": 163587,
    "vocab": {
        token: rank for token, rank in vocab.items()
        if token and any(token in piece for piece in literal_wire)
    },
    "added_tokens": [
        token for token in full["added_tokens"]
        if token["content"] in ("<|end_of_msg|>", "<|open|>")
    ],
    "pattern": split["pattern"],
}
literal_out = Path("internal/hfbpe/testdata/kimi_k3_literal_reference.json")
literal_out.write_text(json.dumps(literal_fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"literal control fixture {len(literal_fixture['vocab'])} ranks, {literal_out.stat().st_size} bytes")


