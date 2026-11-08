package repoanalysis

import (
	"strings"
	"unicode/utf8"
)

// CitingLines are the lines of text that cite the test name: the name as a whole word, as DeadReferences reads one, so
// a longer name that begins with it cites something else.
func CitingLines(text, name string) []string {
	var lines []string
	for line := range strings.Lines(text) {
		if _, cited := citations(line, name); cited {
			lines = append(lines, strings.TrimRight(line, "\r\n"))
		}
	}
	return lines
}

// FollowRename rewrites each citation of the test old in text to renamed, the name a change gave it, and says how many
// it rewrote: on 2026-11-08 a landing was refused because skill.md cited a test its own change had renamed.
func FollowRename(text, old, renamed string) (string, int) {
	at, cited := citations(text, old)
	if !cited {
		return text, 0
	}
	var rewritten strings.Builder
	last := 0
	for _, start := range at {
		rewritten.WriteString(text[last:start])
		rewritten.WriteString(renamed)
		last = start + len(old)
	}
	rewritten.WriteString(text[last:])
	return rewritten.String(), len(at)
}

// citations are where text cites name as a whole word: no name rune before it or after it.
func citations(text, name string) ([]int, bool) {
	var at []int
	for from := 0; ; {
		found := strings.Index(text[from:], name)
		if found < 0 {
			break
		}
		start, end := from+found, from+found+len(name)
		before, _ := utf8.DecodeLastRuneInString(text[:start])
		after, _ := utf8.DecodeRuneInString(text[end:])
		if (start == 0 || !isNameRune(before)) && (end == len(text) || !isNameRune(after)) {
			at = append(at, start)
		}
		from = end
	}
	return at, len(at) > 0
}
