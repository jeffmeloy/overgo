package tokenizer

import (
	"html"
	"strings"
)

// NormalizeEscapedWidthWhitespace HTML-unescapes twice, folds fullwidth ASCII
// and ideographic space, then collapses whitespace runs.
func NormalizeEscapedWidthWhitespace(text string) string {
	text = html.UnescapeString(html.UnescapeString(text))
	text = strings.Map(func(r rune) rune {
		switch {
		case r >= 0xFF01 && r <= 0xFF5E:
			return r - 0xFEE0
		case r == 0x3000:
			return ' '
		default:
			return r
		}
	}, text)
	return strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
}
