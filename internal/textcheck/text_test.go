package textcheck

import "testing"

func TestBounded(t *testing.T) {
	if !Bounded("valid", 5, "\x00\r\n") || Bounded(" valid", 6, "\x00\r\n") || Bounded("too long", 4, "\x00\r\n") || Bounded("two\nlines", 32, "\x00\r\n") {
		t.Fatal("bounded canonical text policy drifted")
	}
}

func TestBoundedToken(t *testing.T) {
	if !BoundedToken("valid-token.2", 32, "/\\") ||
		BoundedToken("two words", 32, "/\\") ||
		BoundedToken("two\u00a0words", 32, "/\\") ||
		BoundedToken("item/step", 32, "/\\") ||
		BoundedToken("control\u0085", 32, "/\\") {
		t.Fatal("bounded token policy drifted")
	}
}

func TestLowerIdentifier(t *testing.T) {
	if !LowerIdentifier("stage_2.output", 32) || LowerIdentifier("Stage", 32) || LowerIdentifier("two words", 32) || LowerIdentifier("stage", 4) {
		t.Fatal("lower identifier policy drifted")
	}
}
