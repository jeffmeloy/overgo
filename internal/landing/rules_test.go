package landing

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// TestNewLegacyFormsNamesOnlyWhatAChangeAdds refuses the three forms go fix
// leaves when a change adds them, and passes a file that already held them.
func TestNewLegacyFormsNamesOnlyWhatAChangeAdds(t *testing.T) {
	t.Parallel()
	const held = `package p

import (
	"sort"
	"sync"
)

var once sync.Once

func Order(values []int) { sort.Slice(values, func(i, j int) bool { return values[i] < values[j] }) }
`
	const added = held + `
func Name(name string) string {
	if name == "" {
		name = "default"
	}
	return name
}

func Count(count int) int {
	if 0 == count {
		count = 3
	}
	return count
}

func Stable(values []int) { sort.SliceStable(values, func(i, j int) bool { return values[i] < values[j] }) }

var twice sync.Once

func Keep(name, other string) string {
	if name == "" {
		other = "x"
	}
	return other
}
`
	snapshot := func(source string) gosyntax.Snapshot {
		t.Helper()
		result, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	if findings := newLegacyForms(snapshot(held), snapshot(held)); len(findings) != 0 {
		t.Fatalf("an unchanged file was refused: %v", findings)
	}
	findings := newLegacyForms(snapshot(held), snapshot(added))
	var scopes []string
	for _, finding := range findings {
		scopes = append(scopes, finding[strings.LastIndex(finding, " ")+1:])
	}
	slices.Sort(scopes)
	if want := []string{"Count", "Name", "Stable", "package"}; !slices.Equal(scopes, want) {
		t.Fatalf("findings = %q, want one each in %v", findings, want)
	}
}
