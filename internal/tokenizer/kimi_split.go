package tokenizer

import "unicode"

// A zero-length regex result leaves input in the Isolated Split remainder.
const kimiNoMatch = 0

// Kimi's declared Split expression has ordered alternatives: Han, two
// non-Han letter forms, numbers, punctuation, and whitespace. Keep the
// scanner shared by native and Hugging Face byte-level consumers.
func preTokenizeKimi(text string) []string {
	return rxIsolatedSplit([]string{text}, kimiMatch)
}

// Script=Han ranges are pinned to the Colibri K3 tokenizer implementation.
const kimiHanClass = "\u2E80-\u2E99\u2E9B-\u2EF3\u2F00-\u2FD5\u3005\u3007\u3021-\u3029\u3038-\u303B\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFA6D\uFA70-\uFAD9\U00016FE2-\U00016FE3\U00016FF0-\U00016FF1\U00020000-\U0002A6DF\U0002A700-\U0002B739\U0002B740-\U0002B81D\U0002B820-\U0002CEA1\U0002CEB0-\U0002EBE0\U0002EBF0-\U0002EE5D\U0002F800-\U0002FA1D\U00030000-\U0003134A\U00031350-\U000323AF"

var kimiHanRanges = parseRuneRanges(kimiHanClass)

func kimiHan(r rune) bool {
	if r < kimiHanRanges[0][0] {
		return false
	}
	low, high := 0, len(kimiHanRanges)-1
	for low <= high {
		middle := (low + high) / 2
		span := kimiHanRanges[middle]
		if r < span[0] {
			high = middle - 1
		} else if r > span[1] {
			low = middle + 1
		} else {
			return true
		}
	}
	return false
}

// Go's Unicode 15 tables miss these Unicode 16 cased letters. The other
// pinned HF letter deltas are uncased and belong to both Kimi intersections.
// Categories are from Unicode 16 UnicodeData.txt.
func kimiNewUpper(r rune) bool {
	switch r {
	case '\u1C89', '\uA7CB', '\uA7CC', '\uA7DA', '\uA7DC':
		return true
	}
	return false
}

func kimiNewLower(r rune) bool {
	switch r {
	case '\u1C8A', '\uA7CD', '\uA7DB':
		return true
	}
	return false
}

func kimiNewUncased(r rune) bool {
	if r < hfUnicodeDeltaL[0][0] {
		return false
	}
	return rxHFDelta("L", r) && !kimiNewUpper(r) && !kimiNewLower(r)
}

func kimiUpper(r rune) bool {
	return !kimiHan(r) && (unicode.Is(unicode.Lu, r) || unicode.Is(unicode.Lt, r) ||
		unicode.Is(unicode.Lm, r) || unicode.Is(unicode.Lo, r) || rxMark(r) ||
		kimiNewUpper(r) || kimiNewUncased(r))
}

func kimiLower(r rune) bool {
	return !kimiHan(r) && (unicode.Is(unicode.Ll, r) || unicode.Is(unicode.Lm, r) ||
		unicode.Is(unicode.Lo, r) || rxMark(r) || kimiNewLower(r) || kimiNewUncased(r))
}

func kimiWord(v []rune, start int, firstAlternative bool) int {
	if start >= len(v) {
		return kimiNoMatch
	}
	end := start
	for end < len(v) && kimiUpper(v[end]) {
		end++
	}
	if firstAlternative {
		// The first expression greedily takes upper-category runes, then
		// backtracks until at least one lower-category rune can match.
		for split := end; split >= start; split-- {
			if split >= len(v) || !kimiLower(v[split]) {
				continue
			}
			last := split + 1
			for last < len(v) && kimiLower(v[last]) {
				last++
			}
			return last - start + contractionLength(v[last:], true)
		}
		return kimiNoMatch
	}
	if end == start {
		return kimiNoMatch
	}
	for end < len(v) && kimiLower(v[end]) {
		end++
	}
	return end - start + contractionLength(v[end:], true)
}

func kimiMatch(v []rune, i int) int {
	if kimiHan(v[i]) {
		end := i + 1
		for end < len(v) && kimiHan(v[end]) {
			end++
		}
		return end - i
	}
	// The optional prefix excludes CR/LF, letters, and numbers. Regex
	// greediness tries that prefix before the no-prefix branch.
	prefix := v[i] != '\r' && v[i] != '\n' && !rxLetter(v[i]) && !rxNumber(v[i])
	for _, firstAlternative := range [...]bool{true, false} {
		if prefix {
			if length := kimiWord(v, i+1, firstAlternative); length > 0 {
				return length + 1
			}
		}
		if length := kimiWord(v, i, firstAlternative); length > 0 {
			return length
		}
	}
	if rxNumber(v[i]) {
		return rxNumberMatch(v, i)
	}
	start := i
	if v[i] == ' ' {
		start++
	}
	if start < len(v) && !rxWhite(v[start]) && !rxLetter(v[start]) && !rxNumber(v[start]) {
		end := start + 1
		for end < len(v) && !rxWhite(v[end]) && !rxLetter(v[end]) && !rxNumber(v[end]) {
			end++
		}
		for end < len(v) && (v[end] == '\r' || v[end] == '\n') {
			end++
		}
		return end - i
	}
	if rxWhite(v[i]) {
		end, lastNewline := i, i
		sawNewline := false
		for end < len(v) && rxWhite(v[end]) {
			if v[end] == '\r' || v[end] == '\n' {
				lastNewline = end + 1
				sawNewline = true
			}
			end++
		}
		if sawNewline {
			return lastNewline - i
		}
		if end < len(v) && end-i > 1 {
			return end - i - 1
		}
		return end - i
	}
	// Isolated Split retains any rune no declared alternative matched.
	return kimiNoMatch
}
