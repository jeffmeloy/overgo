package tokenizer

import (
	"cmp"
	"slices"
	"strings"
	"unicode/utf8"
)

// NormalizeNFC applies Unicode 16 canonical composition. Invalid UTF-8 bytes
// are preserved verbatim as normalization boundaries. It does not apply
// compatibility folding, strip accents, or change the legacy WPM normalizer.
// Ordering and composition follow Unicode 16 sections 3.11 and 3.12;
// the pinned Colibri normalization port and data are documented in NFC.md.
func NormalizeNFC(text string) string {
	if strings.IndexFunc(text, func(r rune) bool { return r >= utf8.RuneSelf }) < 0 {
		return text
	}
	values := make([]rune, 0, len(text))
	for position := 0; position < len(text); {
		character, width := utf8.DecodeRuneInString(text[position:])
		if character == utf8.RuneError && width == 1 {
			// Negative runes are disjoint from valid Unicode scalar values.
			values = append(values, -1-rune(text[position]))
		} else if decomposition, ok := nfdTable[character]; ok {
			values = append(values, decomposition...)
		} else if decomposition, ok := nfcDecompositionDelta[character]; ok {
			values = append(values, decomposition...)
		} else {
			values = append(values, character)
		}
		position += width
	}
	// Stable sorting retains equal-class order, without insertion sort's
	// quadratic cost for arbitrarily long alternating combining-mark runs.
	for start := 0; start < len(values); {
		marks := start
		if nfcCombiningClass[values[marks]] == 0 {
			marks++
		}
		end := marks
		for end < len(values) && nfcCombiningClass[values[end]] != 0 {
			end++
		}
		slices.SortStableFunc(values[marks:end], func(a, b rune) int {
			return cmp.Compare(nfcCombiningClass[a], nfcCombiningClass[b])
		})
		start = end
	}
	// Compact in place. A combining mark blocks composition when its class
	// is equal to or greater than the current class; a starter also blocks.
	starter, written := 0, 1
	previousClass := nfcCombiningClass[values[0]]
	for _, character := range values[1:] {
		class := nfcCombiningClass[character]
		if previousClass == 0 || previousClass < class {
			if composite, ok := nfcCompose(values[starter], character); ok {
				values[starter] = composite
				continue
			}
		}
		if class == 0 {
			starter = written
		}
		values[written] = character
		written++
		previousClass = class
	}
	var output strings.Builder
	output.Grow(len(text))
	for _, character := range values[:written] {
		if character < 0 {
			output.WriteByte(byte(-1 - character))
		} else {
			output.WriteRune(character)
		}
	}
	return output.String()
}

// Hangul constants and arithmetic are normative Unicode 16 section 3.12:
// https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-3/
const (
	nfcHangulSBase  = 0xAC00
	nfcHangulLBase  = 0x1100
	nfcHangulVBase  = 0x1161
	nfcHangulTBase  = 0x11A7
	nfcHangulLCount = 19
	nfcHangulVCount = 21
	nfcHangulTCount = 28
	nfcHangulNCount = nfcHangulVCount * nfcHangulTCount
	nfcHangulSCount = nfcHangulLCount * nfcHangulNCount
)

func nfcCompose(first, second rune) (rune, bool) {
	if first >= nfcHangulLBase && first < nfcHangulLBase+nfcHangulLCount &&
		second >= nfcHangulVBase && second < nfcHangulVBase+nfcHangulVCount {
		return nfcHangulSBase + ((first-nfcHangulLBase)*nfcHangulVCount+second-nfcHangulVBase)*nfcHangulTCount, true
	}
	if first >= nfcHangulSBase && first < nfcHangulSBase+nfcHangulSCount &&
		(first-nfcHangulSBase)%nfcHangulTCount == 0 && second > nfcHangulTBase && second < nfcHangulTBase+nfcHangulTCount {
		return first + second - nfcHangulTBase, true
	}
	value, ok := nfcComposition[[2]rune{first, second}]
	return value, ok
}
