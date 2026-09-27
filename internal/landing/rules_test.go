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

// TestNewPolicyLiteralsIgnoresRenames refuses a number a change adds and
// passes one a change only carries into a renamed function.
func TestNewPolicyLiteralsIgnoresRenames(t *testing.T) {
	t.Parallel()
	snapshot := func(source string) gosyntax.Snapshot {
		t.Helper()
		result, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte("package p\n\n" + source)})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	base := snapshot("func half(n int) int { return n / 2 }\n")
	if findings := newPolicyLiterals(base, snapshot("func Half(n int) int { return n / 2 }\n")); len(findings) != 0 {
		t.Fatalf("a renamed function's number was refused: %v", findings)
	}
	if findings := newPolicyLiterals(base, snapshot("func Half(n int) int { return n / 2 }\n\nfunc third(n int) int { return n / 2 / 2 }\n")); len(findings) != 1 {
		t.Fatalf("an added number was not refused once: %v", findings)
	}
}

// TestRanPackagesNamesOnlyWhatRan counts the packages go test ran, slowest
// first, and leaves out the ones it replayed from its cache.
func TestRanPackagesNamesOnlyWhatRan(t *testing.T) {
	t.Parallel()
	output := strings.Join([]string{
		`{"Action":"output","Package":"overgo/internal/cached","Output":"ok  \tovergo/internal/cached\t(cached)\n"}`,
		`{"Action":"pass","Package":"overgo/internal/cached","Elapsed":0.002}`,
		`{"Action":"pass","Package":"overgo/internal/fast","Elapsed":2}`,
		`{"Action":"pass","Package":"overgo/internal/fast","Test":"TestInner","Elapsed":1}`,
		`{"Action":"pass","Package":"overgo/internal/slow","Elapsed":300}`,
	}, "\n")
	if got, want := ranPackages(output), "ran 2 packages: internal/slow 300s, internal/fast 2s"; got != want {
		t.Fatalf("ranPackages = %q, want %q", got, want)
	}
}
