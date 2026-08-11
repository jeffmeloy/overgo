package tokenizer

import "strings"

// nfdString applies canonical decomposition (Unicode NFD) per rune using
// nfdTable. It intentionally OMITS canonical-combining-class reordering.
//
// The sole caller, preprocessWPM, applies this only when StripAccents is set and
// immediately deletes every combining mark on the very next lines:
//
//	if v.StripAccents && unicode.Is(unicode.M, character) {
//		continue
//	}
//
// (see wpm.go). Because all marks are discarded, the RELATIVE ORDER of marks
// produced by decomposition has zero observable effect, so a per-rune recursive
// canonical decomposition is byte-for-byte equivalent to full-string NFD once
// the marks are stripped. See nfd_test.go for the parity gate proving this
// against golang.org/x/text over a broad Unicode corpus.
//
// The table entries are already fully (recursively) expanded — norm.NFD emits
// the terminal decomposition — so a single table lookup per rune suffices.
func nfdString(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if d, ok := nfdTable[r]; ok {
			for _, c := range d {
				b.WriteRune(c)
			}
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}
