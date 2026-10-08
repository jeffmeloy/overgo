package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestPastHeldoutVerdictsQuery drives overgodb query's -verdicts flag over a
// store that holds no verdict: the listing says so with the sequence it read
// through, the journal's own or the one -to-sequence names, and a sequence the
// journal has not reached is refused with nothing printed. The read moves no
// journal head.
func TestPastHeldoutVerdictsQuery(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindEvidence, "an earlier record"))
	_, earlier := store.Head()
	testutil.PublishArtifact(t, store, testutil.ArtifactID(t, artifact.KindEvidence, "a later record"))
	head, sequence := store.Head()
	for _, through := range []uint64{0, earlier, sequence} {
		arguments := []string{"-repo", root, "-verdicts"}
		want := sequence
		if through != 0 {
			arguments, want = append(arguments, "-to-sequence", fmt.Sprint(through)), through
		}
		var output bytes.Buffer
		if err := runQuery(arguments, &output); err != nil {
			t.Fatal(err)
		}
		if got := strings.TrimSpace(output.String()); got != fmt.Sprintf("0 held-out verdict(s) through sequence %d; 0 bytes of verdict documents", want) {
			t.Fatalf("%v printed %q, want no verdict through sequence %d", arguments, got, want)
		}
	}
	var output bytes.Buffer
	if err := runQuery([]string{"-repo", root, "-verdicts", "-to-sequence", fmt.Sprint(sequence + 1)}, &output); err == nil || output.Len() != 0 {
		t.Fatalf("a sequence the journal has not reached printed %q (%v)", output.String(), err)
	}
	if current, currentSequence := store.Head(); current != head || currentSequence != sequence {
		t.Fatal("the read-only listing moved the journal")
	}
}
