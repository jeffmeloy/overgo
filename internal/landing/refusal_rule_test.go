package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/repoanalysis"
)

// TestARefusalRecordsTheRuleThatRefused holds a refusal of the rules stage to naming every rule it broke, read from the
// text each rule's own check writes, so a rule whose refusals are not defects can be counted and found; a refusal of
// another stage names none.
func TestARefusalRecordsTheRuleThatRefused(t *testing.T) {
	t.Parallel()
	snapshot := func(source string) gosyntax.Snapshot {
		t.Helper()
		result, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	base := snapshot("// Package p is a fixture.\npackage p\n")
	candidate := snapshot(`// Package p is a fixture.
package p

import (
	"sort"
	"sync"
)

// Sorts values and counts a few.
func Exported(values []int) int {
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	var once sync.Once
	once.Do(func() {})
	return 7
}
`)
	style := repoanalysis.ValidateGoStyleDelta(candidate, base)
	if style == nil {
		t.Fatal("the fixture breaks no style rule")
	}
	refused := append([]string{pinsRefusal([]string{"docs/verification/record.json"}), style.Error()}, newPolicyLiterals(base, candidate)...)
	refused = append(append(refused, newLegacyForms(base, candidate)...),
		exportRefusal("p.Exported", "internal/p"), uncoveredRefusal("internal/p"), countRefusal(9, "row/do", 3))
	got := Classify("row/do", "gate: "+stagePrefix+"rules: "+strings.Join(refused, "; "))
	if want := []string{"pins", "style", "inline-numbers", "legacy-forms", "exports", "cuda-lanes", "count"}; got.Class != "rules" || !slices.Equal(got.Rules, want) {
		t.Fatalf("a refusal of every rule classed %s naming %v, want rules naming %v", got.Class, got.Rules, want)
	}
	if count := Classify("row/do", "gate: "+stagePrefix+"rules: "+countRefusal(9, "row/do", 3)); !slices.Equal(count.Rules, []string{"count"}) {
		t.Fatalf("a refusal of the count alone named %v", count.Rules)
	}
	if other := Classify("row/do", "gate: "+stagePrefix+"test: FAIL internal/p"); other.Class != "test" || len(other.Rules) != 0 {
		t.Fatalf("a refusal of the test stage classed %s naming %v, want test naming none", other.Class, other.Rules)
	}
}
