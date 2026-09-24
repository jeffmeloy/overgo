package tokenizer

import "unicode"

// These Unicode 16 additions are measured against the Hugging Face tokenizers
// regex engine used by the pinned RxBrain artifact; Go's tables are Unicode 15.
func rxHFDelta(class string, r rune) bool {
	var spans [][2]rune
	switch class {
	case "L":
		spans = hfUnicodeDeltaL[:]
	case "M":
		spans = hfUnicodeDeltaM[:]
	case "N":
		spans = hfUnicodeDeltaN[:]
	case "P":
		spans = hfUnicodeDeltaP[:]
	case "S":
		spans = hfUnicodeDeltaS[:]
	}
	for _, span := range spans {
		if span[0] <= r && r <= span[1] {
			return true
		}
	}
	return false
}

func rxLetter(r rune) bool { return unicode.IsLetter(r) || rxHFDelta("L", r) }
func rxMark(r rune) bool   { return unicode.Is(unicode.M, r) || rxHFDelta("M", r) }
func rxNumber(r rune) bool { return unicode.IsNumber(r) || rxHFDelta("N", r) }
func rxPunct(r rune) bool  { return unicode.Is(unicode.P, r) || rxHFDelta("P", r) }
func rxSymbol(r rune) bool { return unicode.Is(unicode.S, r) || rxHFDelta("S", r) }
func rxWhite(r rune) bool  { return unicode.IsSpace(r) }

var rxCJKRanges = parseRuneRanges(rxCJKClass)

func rxJapanese(r rune) bool { return runeInRanges(r, rxCJKRanges) }
func rxASCIIPunct(r rune) bool {
	return r >= 33 && r <= 47 || r >= 58 && r <= 64 || r >= 91 && r <= 96 || r >= 123 && r <= 126
}
func rxASCIILetter(r rune) bool { return r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' }
func rxNumberMatch(v []rune, i int) int {
	if !rxNumber(v[i]) {
		return 0
	}
	j := i
	for j < len(v) && rxNumber(v[j]) && j-i < 3 {
		j++
	}
	return j - i
}
func rxCJKMatch(v []rune, i int) int {
	if !rxJapanese(v[i]) {
		return 0
	}
	j := i
	for j < len(v) && rxJapanese(v[j]) {
		j++
	}
	return j - i
}
func rxUnicodeMatch(v []rune, i int) int {
	n := len(v)
	// ASCII punctuation followed by ASCII letters.
	if rxASCIIPunct(v[i]) && i+1 < n && rxASCIILetter(v[i+1]) {
		j := i + 2
		for j < n && rxASCIILetter(v[j]) {
			j++
		}
		return j - i
	}
	// Optional non-newline, non-letter, non-punctuation/symbol prefix, then letters/marks.
	start := i
	if v[i] != '\r' && v[i] != '\n' && !rxLetter(v[i]) && !rxPunct(v[i]) && !rxSymbol(v[i]) && i+1 < n && (rxLetter(v[i+1]) || rxMark(v[i+1])) {
		start++
	}
	if rxLetter(v[start]) || rxMark(v[start]) {
		j := start + 1
		for j < n && (rxLetter(v[j]) || rxMark(v[j])) {
			j++
		}
		return j - i
	}
	// Optional ASCII space, punctuation/symbol run, then CR/LF run.
	start = i
	if v[i] == ' ' && i+1 < n && (rxPunct(v[i+1]) || rxSymbol(v[i+1])) {
		start++
	}
	if rxPunct(v[start]) || rxSymbol(v[start]) {
		j := start + 1
		for j < n && (rxPunct(v[j]) || rxSymbol(v[j])) {
			j++
		}
		for j < n && (v[j] == '\r' || v[j] == '\n') {
			j++
		}
		return j - i
	}
	// Greedy whitespace up to its last newline.
	if rxWhite(v[i]) {
		j := i
		lastNewline := 0
		for j < n && rxWhite(v[j]) {
			if v[j] == '\r' || v[j] == '\n' {
				lastNewline = j + 1
			}
			j++
		}
		if lastNewline > i {
			return lastNewline - i
		}
		if j == n {
			return j - i
		}
		if j-i > 1 {
			return j - i - 1
		}
		return j - i
	}
	return 0
}
func rxIsolatedSplit(parts []string, match func([]rune, int) int) []string {
	var output []string
	for _, part := range parts {
		r := []rune(part)
		start := 0
		for i := 0; i < len(r); {
			m := match(r, i)
			if m == 0 {
				i++
				continue
			}
			if start < i {
				output = append(output, string(r[start:i]))
			}
			output = append(output, string(r[i:i+m]))
			i += m
			start = i
		}
		if start < len(r) {
			output = append(output, string(r[start:]))
		}
	}
	return output
}

func preTokenizeRxNumber(text string) []string { return rxIsolatedSplit([]string{text}, rxNumberMatch) }
func preTokenizeRxCJK(text string) []string    { return rxIsolatedSplit([]string{text}, rxCJKMatch) }
func preTokenizeRxUnicode(text string) []string {
	return rxIsolatedSplit([]string{text}, rxUnicodeMatch)
}
