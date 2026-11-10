package sampling

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestTheGreedyShortcutRefusesLogitsOfAnotherSize holds the greedy draw under a grammar to the full path's admission
// (SQA-357): logits shorter or longer than the grammar's vocabulary are refused, as applyGrammar refuses them, and the
// grammar does not advance, so a draw of the right size after them takes the grammar's first token as a fresh sampler
// does. A six-token vocabulary whose grammar takes x or y once took four logits peaked at x on the shortcut.
func TestTheGreedyShortcutRefusesLogitsOfAnotherSize(t *testing.T) {
	t.Parallel()
	pieces := [][]byte{[]byte("a"), []byte("b"), []byte("x"), []byte("y"), []byte("z"), nil}
	vocabulary := testcheck.Must(NewGBNFVocabulary(pieces, []int{len(pieces) - 1}, nil))(t)
	grammar := testcheck.Must(vocabulary.Compile(`root ::= [xy]+`, "root", GBNFLazyOptions{}))(t)
	sampler := testcheck.Must(New(Config{GBNF: grammar}))(t)
	for _, logits := range [][]float32{{0, 0, 2, 0}, {0, 0, 2, 0, 0, 0, 0, 0}} {
		if token, err := sampler.Sample(logits); err == nil {
			t.Errorf("a greedy draw over %d logits for a %d-token vocabulary took token %d", len(logits), len(pieces), token)
		}
	}
	// A refused draw leaves the grammar where it was: y, the likeliest, is still a first token it takes.
	if token := testcheck.Must(sampler.Sample([]float32{0, 0, 1, 2, 0, 0}))(t); token != 3 {
		t.Errorf("after the refused draws the grammar took token %d, want y (3)", token)
	}
}
