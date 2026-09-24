package tokenizer

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

// The Unicode Consortium corpus was frozen before candidate implementation;
// expectations come from its columns, never from our generated tables.
func TestNFCUnicode16Conformance(t *testing.T) {
	raw, err := os.ReadFile("testdata/NormalizationTest-16.0.0.txt")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "d811971453e7075e1ad56fb1b301eece5aa80757b81f6156e74a1bfb3ae5ceb1" {
		t.Fatal("independent Unicode16 corpus identity changed", got)
	}
	listed := map[rune]bool{}
	part, rows, lineNumber := "", 0, 0
	for line := range strings.SplitSeq(string(raw), "\n") {
		lineNumber++
		line, _, _ = strings.Cut(line, "#")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "@") {
			part = line
			continue
		}
		fields := strings.Split(line, ";")
		if len(fields) != 6 {
			t.Fatalf("line %d: unexpected corpus columns", lineNumber)
		}
		columns := make([]string, 5)
		for index := range columns {
			var value strings.Builder
			for encoded := range strings.FieldsSeq(fields[index]) {
				scalar, err := strconv.ParseInt(encoded, 16, 32)
				if err != nil || !utf8.ValidRune(rune(scalar)) {
					t.Fatalf("line %d: invalid scalar %s", lineNumber, encoded)
				}
				value.WriteRune(rune(scalar))
			}
			columns[index] = value.String()
		}
		if part == "@Part1" {
			scalars := []rune(columns[0])
			if len(scalars) != 1 {
				t.Fatalf("line %d: Part1 source is not one scalar", lineNumber)
			}
			listed[scalars[0]] = true
		}
		for index, expected := range []string{columns[1], columns[1], columns[1], columns[3], columns[3]} {
			if actual := NormalizeNFC(columns[index]); actual != expected {
				t.Fatalf("line %d column %d: NFC(%U)=%U want %U", lineNumber, index+1, []rune(columns[index]), []rune(actual), []rune(expected))
			}
		}
		rows++
	}
	if rows != 19965 {
		t.Fatalf("corpus rows=%d", rows)
	}
	unchanged := 0
	for scalar := rune(0); scalar <= utf8.MaxRune; scalar++ {
		if !utf8.ValidRune(scalar) || listed[scalar] {
			continue
		}
		input := string(scalar)
		if actual := NormalizeNFC(input); actual != input {
			t.Fatalf("unlisted scalar %U changed to %U", scalar, []rune(actual))
		}
		unchanged++
	}
	t.Logf("Unicode16: %d corpus rows, %d NFC column checks, %d unlisted scalar invariants", rows, rows*5, unchanged)
}

func TestNFCBoundaryAndLongCombiningRun(t *testing.T) {
	for _, example := range []struct{ input, expected string }{
		{"", ""}, {"plain\x00bytes", "plain\x00bytes"},
		{"caf\u0065\u0301", "caf\u00e9"},
		{"A\u0301\u0327", "\u00c1\u0327"},
		{"\u1100\u1161\u11a8", "\uac01"},
		{"\u0344", "\u0308\u0301"},
		{"\ufb01", "\ufb01"}, {"\u212b", "\u00c5"},
		{"e\xff\u0301", "e\xff\u0301"},
		{"\xffe\u0301", "\xff\u00e9"},
		{"\xed\xa0\x80A\u0301", "\xed\xa0\x80\u00c1"},
		{"e\x00\u0301", "e\x00\u0301"},
	} {
		if got := NormalizeNFC(example.input); got != example.expected {
			t.Fatalf("NFC(%q)=%q want %q", example.input, got, example.expected)
		}
	}
	// Pinned Colibri regression uses 32768 pairs to expose insertion sorting.
	const pairs = 32768
	input := "A" + strings.Repeat("\u0301\u0327", pairs)
	expected := "\u00c1" + strings.Repeat("\u0327", pairs) + strings.Repeat("\u0301", pairs-1)
	if got := NormalizeNFC(input); got != expected || NormalizeNFC(got) != got {
		t.Fatal("long alternating combining run lost ordering, composition or idempotence")
	}
}
