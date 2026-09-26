package codeprofile

import (
	"slices"
	"testing"
)

// TestConsumerReferencesSortByIdentityOffsetKind holds the consumer graph's
// reference order to its key -- the (from, to) declaration identity, then
// the site offset, then the relation kind -- which the manifest's bytes and
// identity depend on.
func TestConsumerReferencesSortByIdentityOffsetKind(t *testing.T) {
	t.Parallel()
	a := ConsumerDeclaration{Package: "p", File: "p/a.go", Kind: "function", Name: "A"}
	b := ConsumerDeclaration{Package: "p", File: "p/b.go", Kind: "function", Name: "B"}
	want := []ConsumerReference{
		{From: a, To: b, Kind: "call", Offset: 5},
		{From: a, To: b, Kind: "call", Offset: 9},
		{From: a, To: b, Kind: "use", Offset: 9},
		{From: b, To: a, Kind: "call", Offset: 1},
	}
	got := []ConsumerReference{want[3], want[2], want[1], want[0]}
	sortConsumerReferences(got)
	if !slices.Equal(got, want) {
		t.Fatalf("sorted references = %+v, want %+v", got, want)
	}
}
