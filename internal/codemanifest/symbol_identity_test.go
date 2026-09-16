package codemanifest

import (
	"bytes"
	"cmp"
	"encoding/json"
	"slices"
	"testing"
)

func TestTypedSymbolIdentity(t *testing.T) {
	symbols := []SymbolID{
		{"p", "c", "", "N", SymbolFunction},
		{"p0", "c", "", "N", SymbolFunction},
		{"p/a", "c", "", "N", SymbolFunction},
		{"p", "c0", "", "N", SymbolFunction},
		{"p", "c/a", "", "N", SymbolFunction},
		{"p", "c", "R", "N", SymbolFunction},
		{"p", "c", "R0", "N", SymbolFunction},
		{"p", "c", "R", "N0", SymbolFunction},
		{"p", "c", "R", "N", SymbolMethod},
		{"p", "c", "\x01", "N", SymbolFunction},
		{"p", "c", "é", "N", SymbolFunction},
	}
	t.Run("legacy order and equality", func(t *testing.T) {
		for _, left := range symbols {
			if err := validateSymbolID(left); err != nil {
				t.Fatal(err)
			}
			for _, right := range symbols {
				want := cmp.Compare(legacySymbolIdentity(left), legacySymbolIdentity(right))
				if got := compareSymbolID(left, right); got != want || (left == right) != (want == 0) {
					t.Fatalf("symbol order/equality changed: %+v %+v: got %d want %d", left, right, got, want)
				}
			}
		}
		optional := []*SymbolID{nil}
		for index := range symbols {
			optional = append(optional, &symbols[index])
		}
		for _, left := range optional {
			for _, right := range optional {
				var beforeLeft, beforeRight string
				if left != nil {
					beforeLeft = legacySymbolIdentity(*left)
				}
				if right != nil {
					beforeRight = legacySymbolIdentity(*right)
				}
				if got, want := compareUncertainty(Uncertainty{Symbol: left}, Uncertainty{Symbol: right}), cmp.Compare(beforeLeft, beforeRight); got != want {
					t.Fatalf("optional symbol order changed: got %d want %d", got, want)
				}
			}
		}
	})
	t.Run("no temporary sort keys", func(t *testing.T) {
		var comparison int
		allocations := testing.AllocsPerRun(1, func() {
			for _, left := range symbols {
				for _, right := range symbols {
					comparison ^= compareSymbolID(left, right)
				}
			}
		})
		if allocations != 0 {
			t.Fatalf("symbol comparisons allocated %g temporary keys", allocations)
		}
		t.Logf("comparison accumulator: %d", comparison)
	})
	t.Run("canonical bytes and read-only validation", func(t *testing.T) {
		value, err := codec.New(fixtureManifest())
		if err != nil {
			t.Fatal(err)
		}
		// Captured from a7a81b14 before changing identity mechanics.
		const legacyIdentity = "profile:sha256:7bf6eb995e366b5796d4eda9f3df11041a329272d87d68f0ab3b460c1fcb5d2c"
		if value.ID.String() != legacyIdentity {
			t.Fatalf("canonical bytes changed: %s", value.ID)
		}
		for _, mutation := range []string{"none", "files", "symbols", "tags"} {
			t.Run(mutation, func(t *testing.T) {
				candidate := clone(value)
				switch mutation {
				case "files":
					slices.Reverse(candidate.Files)
				case "symbols":
					slices.Reverse(candidate.Symbols)
				case "tags":
					candidate.BuildContexts[0].Tags = []string{"cuda", "cuda"}
				}
				before, err := json.Marshal(candidate)
				if err != nil {
					t.Fatal(err)
				}
				validation := candidate.Validate()
				after, err := json.Marshal(candidate)
				if err != nil || !bytes.Equal(before, after) || (validation == nil) != (mutation == "none") {
					t.Fatalf("validation=%v mutated=%v marshal=%v", validation, !bytes.Equal(before, after), err)
				}
			})
		}
	})
}

// Independent legacy oracle: accepted fields cannot contain the NUL separator.
func legacySymbolIdentity(id SymbolID) string {
	return id.Package + "\x00" + id.Context + "\x00" + id.Receiver + "\x00" + id.Name + "\x00" + string(id.Kind)
}
