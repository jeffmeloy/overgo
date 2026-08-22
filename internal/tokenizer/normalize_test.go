package tokenizer

import "testing"

func TestNormalizeEscapedWidthWhitespace(t *testing.T) {
	if got := NormalizeEscapedWidthWhitespace("ａ，　 b"); got != "a, b" {
		t.Fatalf("width normalization = %q", got)
	}
	if got := NormalizeEscapedWidthWhitespace("&amp;amp;  x "); got != "& x" {
		t.Fatalf("escape normalization = %q", got)
	}
}
