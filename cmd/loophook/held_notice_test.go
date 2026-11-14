package main

import (
	"strings"
	"testing"
)

// TestTheHeldNoticeSaysWhatACommitTakes holds the notice a session sent away is given to what a landing commits: the
// files its steps list, the whole tree only where a step lists none (landing's commitScope, since 2026-11-08). It had
// said a commit takes the checkout's whole tree, and on 2026-11-14 the lead held hold-start/do about 55 minutes for a
// helper's unrelated edits on that word, though the landing's candidate held hold-start's four files alone.
func TestTheHeldNoticeSaysWhatACommitTakes(t *testing.T) {
	t.Parallel()
	said := strings.Join(strings.Fields(heldNotice("one")), " ")
	if !strings.Contains(said, "A commit takes the files its steps list, and the whole tree where a step lists none") || strings.Contains(said, "the checkout's whole tree") {
		t.Fatalf("a session sent away is told %q; want it told that a commit takes the files its steps list", said)
	}
}
