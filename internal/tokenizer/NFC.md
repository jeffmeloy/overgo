# Declared canonical normalization

`NormalizeNFC` is the shared Unicode 16 canonical-composition owner. The HF BPE
encoder selects it only for a declared `{"type":"NFC"}` normalizer. Native GGUF
profiles and legacy WPM accent stripping retain their existing behavior.

The implementation reuses all 13,233 existing recursive decompositions in
`nfd_table.go`; they exactly match the pinned Unicode 16 source. A separate
20-entry version delta preserves the older WPM table's semantics. The generated
data adds 934 combining classes and 961 non-Hangul compositions. Hangul uses
the normative arithmetic in [Unicode 16, section 3.12](https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-3/).
Stable ordering handles arbitrarily long combining runs without insertion
sort's quadratic behavior. Composition exclusions come from the pinned table.

The port is based on Colibri `f028d26b422144ed4a69ad9aeaee2553ce0f9572`,
`c/qwen38_nfc.h` and `c/qwen38_nfc_tables.h`. The latter has SHA256
`ae7f710fc7169e7b4f8798124fda348fc829e73aa184d8820489b2e38b0de340`.
The Go port shares Overgo's existing decomposition backing, uses standard
library stable sorting and compacts composition in place. See
`../../licenses/colibri-Apache-2.0.txt` and `../../licenses/Unicode-LICENSE.txt`.

Regenerate from the repository root:

```
go run ./internal/tokenizer/gennfc -source /path/to/colibri/c/qwen38_nfc_tables.h
```

The generator refuses a different source hash or a changed existing
decomposition. It does not modify `nfd_table.go`, install a dependency, or
generate test expectations.
The generator is a buildable Go package covered by normal build and vet checks.

Independent acceptance uses the complete
[Unicode 16 NormalizationTest corpus](https://www.unicode.org/Public/16.0.0/ucd/NormalizationTest.txt),
retained verbatim at `testdata/NormalizationTest-16.0.0.txt` with SHA256
`d811971453e7075e1ad56fb1b301eece5aa80757b81f6156e74a1bfb3ae5ceb1`.
Its 19,965 rows test all five NFC column invariants; the unlisted-scalar
requirement is also checked. This corpus was retained before the implementation.

Invalid UTF-8 bytes are preserved as boundaries by the common normalizer.
The HF NFC encoder rejects invalid UTF-8 input explicitly. Raw added tokens
are extracted before normalization, so normalization cannot compose across a
token boundary or transform the token spelling. NFC currently requires explicit
`normalized:false` added tokens with no stripping or single-word constraints;
unsupported matching semantics refuse encoding rather than being ignored.

Absent/null normalizers and the existing literal space-to-marker mode preserve
their previous encoding behavior. Other normalizer declarations, including
Sequence pipelines, produce an Encode error while allowing decode-only use.
General pre-tokenizer compilation and normalized added-token matching remain
separate obligations; this slice does not claim complete HF pipeline support.
