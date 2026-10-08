package tokenizer

import "unicode"

// deepSeekV3NumberRun is how many digits the rule keeps together: a longer
// number is cut into runs of three from its left.
const deepSeekV3NumberRun = 3

// preTokenizeDeepSeekV3 cuts a text as DeepSeek-V3's tokenizer does, which
// is the rule Unlimited-OCR's tokenizer file holds: three splits, each over
// every piece the one before it left, with what a split does not match kept
// as the pieces between its matches.
//
//	\p{N}{1,3}
//	[一-龥぀-ゟ゠-ヿ]+
//	[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Where it parts from the earlier DeepSeek rule a file may name in its place:
// a number is cut in threes, a mark takes the letters straight after it, and
// marks take the line breaks that follow them.
func preTokenizeDeepSeekV3(text string) []string {
	parts := splitDeepSeekParts([]string{text}, deepSeekV3Number)
	parts = splitDeepSeekParts(parts, deepSeekV3Script)
	return splitDeepSeekParts(parts, deepSeekV3Word)
}

// deepSeekV3Number matches up to three numbers in a row.
func deepSeekV3Number(values []rune, position int) int {
	length := 0
	for length < deepSeekV3NumberRun && position+length < len(values) && unicode.IsNumber(values[position+length]) {
		length++
	}
	return length
}

// deepSeekV3Script matches a run of Chinese characters, hiragana and katakana.
func deepSeekV3Script(values []rune, position int) int {
	end := position
	for end < len(values) && (values[end] >= '一' && values[end] <= '龥' || values[end] >= '぀' && values[end] <= 'ヿ') {
		end++
	}
	return end - position
}

// deepSeekV3Word matches the first of the third split's six alternatives
// that matches at a position, as a regular expression's alternation does.
func deepSeekV3Word(values []rune, position int) int {
	first := values[position]
	run := func(from int, member func(rune) bool) int {
		end := from
		for end < len(values) && member(values[end]) {
			end++
		}
		return end - from
	}
	letter := func(value rune) bool { return unicode.IsLetter(value) || unicode.IsMark(value) }
	mark := func(value rune) bool { return unicode.IsPunct(value) || unicode.IsSymbol(value) }
	lineBreak := func(value rune) bool { return value == '\r' || value == '\n' }
	// An ASCII mark and the ASCII letters straight after it.
	if first <= unicode.MaxASCII && mark(first) {
		if letters := run(position+1, func(value rune) bool { return value <= unicode.MaxASCII && unicode.IsLetter(value) }); letters > 0 {
			return 1 + letters
		}
	}
	// Letters, with one character before them that is no letter, mark or line break.
	if !lineBreak(first) && !unicode.IsLetter(first) && !mark(first) {
		if letters := run(position+1, letter); letters > 0 {
			return 1 + letters
		}
	}
	if letters := run(position, letter); letters > 0 {
		return letters
	}
	// Marks, with a space before them and the line breaks after them.
	from := position
	if first == ' ' {
		from++
	}
	if marks := run(from, mark); marks > 0 {
		return from + marks + run(from+marks, lineBreak) - position
	}
	// Whitespace: up to its last line break; else all but its last character
	// when a word follows, leaving that one to the word; else all of it.
	spaces := run(position, unicode.IsSpace)
	for end := position + spaces; end > position; end-- {
		if lineBreak(values[end-1]) {
			return end - position
		}
	}
	if position+spaces < len(values) && spaces > 1 {
		return spaces - 1
	}
	return spaces
}
