package main

import (
	"testing"

	"overgo/internal/overgodb"
)

// TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession holds the
// dispatch to what it claims: a path that is not a Fractale generation model
// is reported as not one, with nothing opened, so evaluation falls through to
// the model's own runner.
func TestADirectoryThatIsNoFractaleModelOpensNoThoughtBankSession(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	session, opened, err := openThoughtBankSession(t.Context(), store, manifest{}, t.TempDir())
	if session != nil || opened || err != nil {
		t.Fatalf("an empty directory opened (%v, %t, %v), want no session and no error", session, opened, err)
	}
}
