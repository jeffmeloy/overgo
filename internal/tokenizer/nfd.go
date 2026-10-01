package tokenizer

import (
	"strings"
	"sync"
)

// nfdString applies canonical decomposition (Unicode NFD) per rune using
// appendNFD. It intentionally OMITS canonical-combining-class reordering.
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
func nfdString(s string) string {
	values := make([]rune, 0, len(s))
	for _, r := range s {
		values = appendNFD(values, r)
	}
	return string(values)
}

// Hangul constants and arithmetic are normative Unicode 16 section 3.12:
// https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-3/
const (
	hangulSBase  = 0xAC00
	hangulLBase  = 0x1100
	hangulVBase  = 0x1161
	hangulTBase  = 0x11A7
	hangulLCount = 19
	hangulVCount = 21
	hangulTCount = 28
	hangulNCount = hangulVCount * hangulTCount
	hangulSCount = hangulLCount * hangulNCount
)

// appendNFD appends r's full recursive canonical decomposition to values, or
// r itself when it has none. A Hangul syllable decomposes by arithmetic into
// its leading and vowel jamo, then its trailing jamo when it has one; every
// other decomposition is already fully expanded in nfdTable, so a single
// lookup suffices.
func appendNFD(values []rune, r rune) []rune {
	if index := r - hangulSBase; index >= 0 && index < hangulSCount {
		values = append(values, hangulLBase+index/hangulNCount, hangulVBase+index%hangulNCount/hangulTCount)
		if trailing := index % hangulTCount; trailing != 0 {
			values = append(values, hangulTBase+trailing)
		}
		return values
	}
	if decomposition, ok := nfdTable()[r]; ok {
		return append(values, decomposition...)
	}
	return append(values, r)
}

// nfdTable maps each decomposable rune other than a Hangul syllable to its
// decomposition, decoded from nfdPacked once on first use.
var nfdTable = sync.OnceValue(decodeNFD)

// decodeNFD reads nfdPacked's space-terminated records: the first rune of a
// record is the key and the rest is its decomposition.
func decodeNFD() map[rune][]rune {
	table := map[rune][]rune{}
	for record := range strings.FieldsFuncSeq(nfdPacked, func(r rune) bool { return r == ' ' }) {
		values := []rune(record)
		table[values[0]] = values[1:]
	}
	return table
}
