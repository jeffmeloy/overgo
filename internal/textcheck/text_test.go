package textcheck

import "testing"

func TestBounded(t *testing.T) {
	if !Bounded("valid", 5, "\x00\r\n") || Bounded(" valid", 6, "\x00\r\n") || Bounded("too long", 4, "\x00\r\n") || Bounded("two\nlines", 32, "\x00\r\n") {
		t.Fatal("bounded canonical text policy drifted")
	}
}
