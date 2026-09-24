# Declared byte-level pre-tokenization

`hfbpe.Load` compiles a non-null `pre_tokenizer` once. Supported ordered nodes
are `Sequence`, `Digits`, exact recognized `Split` expressions and one final
`ByteLevel`. `internal/tokenizer.CompileBPESplit` resolves those expressions to
the existing native GPT2, Qwen2 and three-digit Llama3 scanners, plus
three exact RxBrain Split stages. Artifact syntax selects the scanner; a model
name never chooses tokenization.

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
The byte alphabet is applied once, after the final ByteLevel node. A partial
Split expression with `Removed/invert:true` is rejected because it would drop
unmatched text; that behavior is accepted only for complete-coverage expressions.
Declared BPE dropout, unknown tokens, nonempty prefixes or suffixes, byte fallback,
unknown fusion and ignored merges are refused until they have exact support.
For the supported no-unknown/no-fallback BPE path, missing initial byte symbols
are dropped before merging, as the pinned Hugging Face engine does.

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
GPT2/Qwen/Llama3 scanners retain their existing Go Unicode tables. The three
RxBrain stages use the pinned Hugging Face regex classes, with explicit deltas
over Go Unicode 15. `testdata/rxbrain-hf-unicode-classes.json` checks L, M, N,
P, S and whitespace on every Unicode scalar. This exact-expression support
does not establish equality for arbitrary regexes or future Unicode versions.
Nonempty normalization Sequences, arbitrary regexes and normalized added-token
matching remain open.

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

## RxBrain declared-path parity

The pinned `Hy-Embodied-RxBrain-1.0/tokenizer.json` has SHA256
`ae5ca95eeb8e9a8774513a996e4e820a05c27db32966aa8e071c10828402cb73`.
It declares an empty normalization Sequence, then isolated number, CJK and
Unicode Split stages, followed by ByteLevel with `use_regex:false`. The third
expression includes punctuation, marks, trailing newlines and whitespace
lookahead. The no-unknown BPE vocabulary omits the carriage-return byte
symbol; discarding it before merges is observable in several reference cases.

`testdata/rxbrain-reference.json` retains 153 independent token-ID and final
piece vectors from Hugging Face tokenizers 0.22.2: 16 curated cases with each
intermediate Split output, 132 seeded held-out cases, and five targeted Unicode
class differences. It has SHA256
`f7dcb5718d7db84575e3d9e208abcc451bcafac737c84c2f5d9f746a85f7c468`.
The earlier deployed legacy scanner differed on 103/153 IDs. The declared
path now matches 153/153 against both the full artifact and a compact
projection. The projection SHA256 is
`133d9763670437cf8c36a6507392f50577d65568f007a0fcc1e7fc11ca5ca61a`;
it keeps the original token IDs and every vocabulary spelling and ranked merge
reachable after the source BPE's missing initial byte symbols are dropped.
Hugging Face reassigns one added-token ID when it loads this sparse projection,
so the independent expected ID stays pinned to the full artifact and the Go
loader checks both. The full Unicode class corpus SHA256 is
`7db1e9a46d0478d3ae4c3d77276cec9c636a8fdb041ff30d2aa6d701d00c0e66`.
The source model's Apache-2.0 notice is in `licenses/RxBrain-Apache-2.0.txt`.

The non-skipped `TestRxBrainProductionParity` passed on the real GPU model:
the declared tokenizer kept the 346-token image/text prompt, 13-token golden
answer prefix and expected stovetop/green-toy answer in both verification and
active serving. This validates that canonical VQA path, not all possible VQA
prompts or changed speech/media output. The existing Krea, Fractale and
Granite identity vectors and 92 independent GPT2/Qwen2 cases also pass.

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
