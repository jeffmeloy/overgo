package clioptions

import (
	"strings"
	"testing"
)

// TestTailWriter proves the bounded writer retains only the last limit bytes
// across many writes and renders the same tail Tail renders over the whole
// stream, so a supervised process's output is diagnosed without being held.
func TestTailWriter(t *testing.T) {
	const limit = 64
	writer := NewTailWriter(limit)
	var whole strings.Builder
	for i := range 5000 {
		chunk := []byte{byte('a' + i%26), byte('0' + i%10), '\n'}
		whole.Write(chunk)
		if _, err := writer.Write(chunk); err != nil {
			t.Fatal(err)
		}
		if writer.Len() > limit {
			t.Fatalf("retention %d exceeded the %d-byte bound at write %d", writer.Len(), limit, i)
		}
	}
	if got, want := writer.Tail(), Tail(whole.String(), limit); got != want {
		t.Fatalf("tail = %q, want %q", got, want)
	}
	if !strings.HasPrefix(writer.Tail(), "...") {
		t.Fatal("a truncated tail must mark the drop")
	}

	small := NewTailWriter(limit)
	small.Write([]byte("short"))
	if small.Tail() != "short" {
		t.Fatalf("an unbounded tail must not prefix: %q", small.Tail())
	}
	small.Reset()
	if small.Len() != 0 || small.Tail() != "" {
		t.Fatal("reset left retained bytes")
	}
}
