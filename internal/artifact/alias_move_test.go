package artifact

import (
	"testing"
)

// TestAliasMoveOwnsTheCompareAndSwap holds the three forms every caller now
// shares: a first binding names no previous target, a move names the one the
// alias held, a removal names the target as both, and the resolving form
// reports no move when the alias is bound to the target already.
func TestAliasMoveOwnsTheCompareAndSwap(t *testing.T) {
	t.Parallel()
	held, err := IdentifyBytes(KindEvidence, []byte("held"))
	if err != nil {
		t.Fatal(err)
	}
	next, err := IdentifyBytes(KindEvidence, []byte("next"))
	if err != nil {
		t.Fatal(err)
	}
	if first := AliasMove("a", next, ID{}); first.Previous != nil || first.Target != next || first.Remove {
		t.Fatalf("first binding = %+v", first)
	}
	if move := AliasMove("a", next, held); move.Previous == nil || *move.Previous != held || move.Target != next {
		t.Fatalf("move = %+v", move)
	}
	if removal := AliasRemoval("a", held); !removal.Remove || removal.Target != held || removal.Previous == nil || *removal.Previous != held {
		t.Fatalf("removal = %+v", removal)
	}
	for _, test := range []struct {
		name  string
		bound ID
		moved bool
	}{
		{"unbound", ID{}, true},
		{"bound elsewhere", held, true},
		{"bound to the target", next, false},
	} {
		binding, moved, err := MoveAlias(t.Context(), codecAliasReader{alias: test.bound}, "a", next)
		if err != nil || moved != test.moved || binding.Target != next {
			t.Fatalf("%s: binding %+v moved=%v err=%v", test.name, binding, moved, err)
		}
		if test.bound.Valid() != (binding.Previous != nil) || binding.Previous != nil && *binding.Previous != test.bound {
			t.Fatalf("%s: previous = %v", test.name, binding.Previous)
		}
	}
}
