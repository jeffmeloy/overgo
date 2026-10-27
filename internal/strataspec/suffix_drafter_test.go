package strataspec

import "testing"

// newDefaultSuffixDrafter is SuffixDrafter() with its defaults: matches of 3 to 32 tokens, half a million tokens.
func newDefaultSuffixDrafter() *SuffixDrafter {
	return NewSuffixDrafter(minimumMatch, defaultMax, defaultTokens)
}

// ProposeAfter is Propose for the history followed by extra (the draft head's proposal for the next tokens, as
// --lookup-chain, which the serve loop does not run, asks), so the lookup continues where those drafts end; the
// history itself is unchanged.
func (d *SuffixDrafter) ProposeAfter(extra []int32, maxK int, out []int32) int {
	if len(extra) == 0 {
		return d.Propose(maxK, out)
	}
	// the index holds the history's trigrams only: appended without indexing, the extra tokens just form the suffix
	n := len(d.history)
	d.history = append(d.history, extra...)
	k := d.Propose(maxK, out)
	d.history = d.history[:n]
	return k
}

// TestSuffixDrafter is Strata's suffix_drafter_test unit tests (src/spec/suffix_drafter_test.cpp).
func TestSuffixDrafter(t *testing.T) {
	t.Run("a repeated passage is proposed verbatim", func(t *testing.T) {
		d := newDefaultSuffixDrafter()
		d.Append(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
		d.Append(10, 11, 12)
		out := make([]int32, 8)
		n := d.Propose(5, out)
		if n != 5 || out[0] != 13 || out[4] != 17 {
			t.Fatalf("repeat proposes %v (%d), where the continuation is 13..17", out[:n], n)
		}
		if d.LastMatch() != 3 {
			t.Fatalf("match length %d, where the shared suffix is 3", d.LastMatch())
		}
	})
	t.Run("no earlier occurrence proposes nothing", func(t *testing.T) {
		d := newDefaultSuffixDrafter()
		d.Append(1, 2, 3, 4, 5, 6, 7)
		if n := d.Propose(4, make([]int32, 4)); n != 0 {
			t.Fatalf("no match proposes %d", n)
		}
	})
	t.Run("the longer match wins though it is older", func(t *testing.T) {
		d := newDefaultSuffixDrafter()
		d.Append(7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3)
		out := make([]int32, 2)
		if n := d.Propose(2, out); n != 2 || out[0] != 100 || out[1] != 101 {
			t.Fatalf("proposes %v, where the longest match continues 100, 101", out[:n])
		}
		if d.LastMatch() != 5 {
			t.Fatalf("match length %d, where the longest is 5", d.LastMatch())
		}
	})
	t.Run("a match shorter than the minimum is ignored", func(t *testing.T) {
		d := NewSuffixDrafter(4, defaultMax, defaultTokens)
		d.Append(1, 2, 3, 9, 5, 2, 3)
		if n := d.Propose(2, make([]int32, 2)); n != 0 {
			t.Fatalf("proposes %d below min_match", n)
		}
	})
	t.Run("periodic text's continuation runs into the suffix", func(t *testing.T) {
		d := newDefaultSuffixDrafter()
		d.Append(1, 2, 3, 1, 2, 3, 1, 2, 3)
		out := make([]int32, 6)
		if n := d.Propose(6, out); n < 3 || out[0] != 1 || out[1] != 2 || out[2] != 3 {
			t.Fatalf("periodic continuation %v", out[:n])
		}
	})
	t.Run("bounded memory past the nominal capacity", func(t *testing.T) {
		d := NewSuffixDrafter(3, 32, 1024)
		for i := range 5000 {
			d.Append(int32(i % 997))
		}
		if n := d.Propose(4, make([]int32, 4)); n <= 0 {
			t.Fatal("no proposal after the nominal capacity overflowed")
		}
	})
	t.Run("propose after continues past the drafts", func(t *testing.T) {
		d := NewSuffixDrafter(3, 32, 256)
		history := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 51, 52, 1, 2}
		d.Append(history...)
		out := make([]int32, 8)
		if k := d.ProposeAfter([]int32{3, 4}, 3, out); k != 3 || out[0] != 5 || out[1] != 6 || out[2] != 7 {
			t.Fatalf("propose_after proposes %v", out[:k])
		}
		if d.LastMatch() < 4 {
			t.Fatalf("propose_after's match length %d does not count the drafts", d.LastMatch())
		}
		if d.Size() != len(history) {
			t.Fatal("propose_after changed the history")
		}
		if k := d.ProposeAfter([]int32{99, 98, 97}, 3, out); k != 0 {
			t.Fatal("propose_after without a match proposes")
		}
		if k := d.Propose(3, out); k != 0 {
			t.Fatal("propose after propose_after proposes")
		}
		if d.ProposeAfter(nil, 3, out) != d.Propose(3, out) {
			t.Fatal("propose_after with no drafts differs from propose")
		}
	})
}
