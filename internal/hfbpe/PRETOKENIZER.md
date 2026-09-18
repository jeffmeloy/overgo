# Declared byte-level pre-tokenization

`hfbpe.Load` compiles a non-null `pre_tokenizer` once. Supported ordered nodes
are `Sequence`, `Digits`, exact recognized `Split` expressions and one final
`ByteLevel`. `internal/tokenizer.CompileBPESplit` resolves those expressions to
the existing native GPT2, Qwen2 and three-digit Llama3 scanners. No model-name
selection or second implementation of those scanners is introduced.

Colibri `f028d26b422144ed4a69ad9aeaee2553ce0f9572`, `c/qwen38.c`, supplied the
boundary-handling comparison: raw added tokens precede normalization, and a
newline match leaves subsequent indentation for the next match. Its scanner
also includes combining-mark classes, so it cannot replace every target's
declared expression. The artifact selects the matching existing Overgo owner.

ByteLevel honors `add_prefix_space` on each incoming segment and `use_regex`
(omission defaults to true). `trim_offsets` is validated but does not affect
this ID-only API. Digits isolates each Unicode number or contiguous numeric
runs as declared. Split supports `Isolated/invert:false` and
`Removed/invert:true` for the recognized complete-coverage expressions.
The byte alphabet is applied once, after the final ByteLevel node.

Semantics were checked against Hugging Face tokenizers commit
`7eb3c7ca428681048507a6c62bcd1337d2386c23`:

- [ByteLevel](https://github.com/huggingface/tokenizers/blob/7eb3c7ca428681048507a6c62bcd1337d2386c23/tokenizers/src/pre_tokenizers/byte_level.rs)
- [Digits](https://github.com/huggingface/tokenizers/blob/7eb3c7ca428681048507a6c62bcd1337d2386c23/tokenizers/src/pre_tokenizers/digits.rs)
- [Split](https://github.com/huggingface/tokenizers/blob/7eb3c7ca428681048507a6c62bcd1337d2386c23/tokenizers/src/pre_tokenizers/split.rs)

Unsupported nodes, regexes, ordering, delimiter behavior and added-token
boundary/normalization flags refuse Encode. Valid decoding remains available.
Supported declared pipelines require explicit raw added tokens (`normalized:
false`), valid UTF-8 and no legacy marker normalizer. NFC still applies after
raw added-token extraction and before pre-tokenization.

Absent/null declarations and `LoadSplit` preserve the established legacy
scanner and token IDs. This compatibility behavior is explicit; it does not
claim absent HF declarations inherently mean Qwen-style splitting. The shared
scanners use Go's Unicode character classes (currently Unicode 15), independently
of NFC's Unicode 16 normalization data. These tests do not establish equality
with every regex engine or future Unicode version. General normalization
Sequences, arbitrary regexes and normalized added-token matching remain open.

## Independent oracle

The two `testdata/*-oracle.json` files retain all 46 inputs and expected ID
vectors from each upstream corpus at llama.cpp commit
`42fc243060709331ff9b158a9ed2cbe37219ae83`. Expected outputs were read directly
from its `models/ggml-vocab-{qwen2,gpt-2}.gguf.out`; neither Overgo nor the
candidate generated them. Windows CRLF is normalized as in the native oracle
test, matching upstream's C++ text-mode input contract.

| Source | SHA256 |
| --- | --- |
| Qwen2 vocabulary | `44c2f46b715f585c6ab513970e8a006bfa5badd6108560054921cf598d154d8c` |
| GPT2 vocabulary | `cedc56ca6e2e89f63e781696d1fd76b4b1d49e6720dee86463e915f6e90016ac` |
| Both input files | `a4f554d42f793610f44fd8e56c97356bc2692d427bf618228ff236059d6a8a19` |
| Qwen2 expected IDs | `04c6278ac3bf07c4af8d80a28f7135fcb664bfc198d9e0845526579c7082d260` |
| GPT2 expected IDs | `619d4283967ae1cee640809b09c31228d919926024094e4b9b99d9727b38269c` |

The vocabulary projection retains original token IDs and every vocabulary
spelling that is a substring of the byte-alphabet-encoded inputs. Every merge
whose joined spelling is such a substring is retained in its original order.
Every reachable intermediate BPE piece is a substring, so this preserves all
possible merges and their relative ranks for these inputs. It is independent
of candidate split boundaries. The projections contain 603/480 vocabulary
entries and 497/374 merges; full-vocabulary comparisons also passed 46/46 each.
The corresponding upstream MIT notice is in `licenses/llama.cpp-MIT.txt`.

Before this change the HF path mismatched 1/46 Qwen2 and 12/46 GPT2 cases;
the native path matched all 92. Substituting only native splitting eliminated
all 13 mismatches before pipeline implementation. The declared pipeline now
matches all 92, including with the projected test fixtures.

## Consumer identities and limits

`testdata/consumer-identity.json` uses the same substring projection, retains
source tokenizer SHA256 values, and pins eight before-change vectors from
commit `3eac92e48aa6b06d73c7364b05616c34086803c9`. These are compatibility
observations, separate from the independent upstream oracle. Cases cover
Krea's telemetry prompt, canonical accent and whitespace, Fractale's retained
code prompt/digits/contractions, and Granite's raw and lowercase frozen
speech-training text from `audioparity/testdata/ctc_training_record.json`.
The Granite vectors also match that existing independent training record.

The real Krea fox-prompt golden (including template, padding and mask) remains
unchanged: IDs SHA256 `b0fe1c0b43f92fc37c3d804791970669f6ac098c60e7ba12f6c82e2b8d527560`,
mask SHA256 `be1a811db02b1ffb5f5bff0fec67336803eb66c5cbca199e23bea4c1f6799c63`.

Granite input `123456789` intentionally changes from nine individual byte IDs
to `[8349,19,14690,22,23,24,25]` under its declared three-digit split. That
observation is not an independent numerical oracle. Changed inputs retain
affected alignment/training/model-output reacquisition obligations. No claim
is made that all deployed prompts are unchanged, that Kimi is complete, or
that model numerical accuracy or throughput has been established by these
tokenizer checks. Nemotron's unsupported encoding pipeline remains decode-only.
