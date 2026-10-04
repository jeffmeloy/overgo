package landing

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/plan"
)

// TestExportUseCountsConsumers holds the export rule to consumers: another
// package's production code that selects a new export through an import of
// its package lets it land under any import name, as does a file that names
// the package but does not parse; another package's test counts only for a
// test-support package, one whose production importers are all test
// support themselves and none a main package. A comment, a string, a local
// that shadows the import, a same-named package from another path and the
// package's own tests leave it refused.
func TestExportUseCountsConsumers(t *testing.T) {
	t.Parallel()
	const exported = "package p\n\nfunc New() int { return 1 }\n"
	const imports = "package q\n\nimport \"overgo/internal/p\"\n\n"
	const tested = imports + "var value = p.New()\n"
	snapshot := func(source string) gosyntax.Snapshot {
		t.Helper()
		result, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte(source)})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	for _, test := range []struct {
		name     string
		files    map[string]string
		consumed bool
	}{
		{"a comment and a string", map[string]string{"internal/q/q.go": "package q\n\n// Built by p.New once the model loads.\nconst note = \"p.New\"\n"}, false},
		{"a shadowing local", map[string]string{"internal/q/q.go": imports + "var other = p.Other\n\nfunc f(p struct{ New int }) int { return p.New }\n"}, false},
		{"a same-named package", map[string]string{"internal/q/q.go": "package q\n\nimport p \"overgo/internal/peer\"\n\nvar value = p.New()\n"}, false},
		{"its own package's test", map[string]string{"internal/p/p_test.go": "package p_test\n\nimport \"overgo/internal/p\"\n\nvar value = p.New()\n"}, false},
		{"only another package's test, of a package production imports", map[string]string{
			"cmd/x/main.go": "package main\n\nimport \"overgo/internal/p\"\n\nfunc main() { p.Other() }\n", "internal/q/q_test.go": tested}, false},
		{"a call", map[string]string{"cmd/x/main.go": "package main\n\nimport \"overgo/internal/p\"\n\nfunc main() { p.New() }\n"}, true},
		{"an aliased import", map[string]string{"internal/q/q.go": "package q\n\nimport made \"overgo/internal/p\"\n\nvar value = made.New()\n"}, true},
		{"a consumer before a local selector", map[string]string{"internal/q/q.go": imports + "func use(...int) {}\nfunc f(local struct{ New int }) { use(p.New(), local.New) }\n"}, true},
		{"a consumer after a local selector", map[string]string{"internal/q/q.go": imports + "func use(...int) {}\nfunc f(local struct{ New int }) { use(local.New, p.New()) }\n"}, true},
		{"another package's test, of a package only tests import", map[string]string{"internal/q/q_test.go": tested}, true},
		{"another package's test, of a package only test support imports", map[string]string{
			"internal/b/b.go":      "package b\n\nimport \"overgo/internal/p\"\n\nvar Other = p.Other\n",
			"internal/c/c_test.go": "package c\n\nimport \"overgo/internal/b\"\n\nvar other = b.Other\n", "internal/q/q_test.go": tested}, true},
		{"an unparsed importer", map[string]string{"internal/q/q.go": imports + "func broken( {\n"}, true},
	} {
		repo := t.TempDir()
		test.files["internal/p/p.go"] = exported
		for name, content := range test.files {
			full := filepath.Join(repo, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		l := &landing{Options: Options{Repo: repo}}
		findings, err := l.newUnconsumedExports(snapshot("package p\n"), snapshot(exported))
		if err != nil || (len(findings) == 0) != test.consumed {
			t.Errorf("%s: findings = %q, %v; consumed = %t", test.name, findings, err, test.consumed)
		}
	}
}

// TestFallbackLintAcceptsLazyReads passes an if that falls back from an
// empty value to a read only the fallback may evaluate: rewritten with
// cmp.Or, the index, dereference, field read, call or receive would run
// even when the value is set.
func TestFallbackLintAcceptsLazyReads(t *testing.T) {
	t.Parallel()
	const lazy = `package p

type config struct{ name string }

func Name(name string, values []string, count *int, fallback *config, load func() string, ready <-chan string) (string, int) {
	if name == "" {
		name = values[0]
	}
	limit := 0
	if limit == 0 {
		limit = *count
	}
	if name == "" {
		name = fallback.name
	}
	if name == "" {
		name = load()
	}
	if name == "" {
		name = <-ready
	}
	return name, limit
}
`
	base, err := (gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte("package p\n")})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := base.Overlay(map[string][]byte{"internal/p/p.go": []byte(lazy)})
	if err != nil {
		t.Fatal(err)
	}
	if findings := newLegacyForms(base, candidate); len(findings) != 0 {
		t.Fatalf("lazy fallbacks were refused: %q", findings)
	}
}

// TestNewLegacyFormsNamesOnlyWhatAChangeAdds refuses the forms go fix leaves
// when a change adds them, and passes a file that already held them.
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
func Stable(values []int) { sort.SliceStable(values, func(i, j int) bool { return values[i] < values[j] }) }

var twice sync.Once
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
	if want := []string{"Stable", "package"}; !slices.Equal(scopes, want) {
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

// TestNewProductionCodeStaysWithinTheRowBudget holds the growth rule to the
// plan's budget: production growth needs a declared reason, deletions pay
// for additions, tests and generated files do not count, and declared nodes
// cap the growth.
func TestNewProductionCodeStaysWithinTheRowBudget(t *testing.T) {
	t.Parallel()
	const small = "package p\n\nfunc A() int { return 1 }\n"
	const large = "package p\n\nfunc A() int { return 1 }\n\nfunc B() int { return A() + A() }\n"
	snapshot := func(files map[string][]byte) gosyntax.Snapshot {
		t.Helper()
		out, err := gosyntax.Snapshot{}.Overlay(files)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	base := snapshot(map[string][]byte{"p/p.go": []byte(small), "p/old.go": []byte(large)})
	grown := productionNodes(snapshot(map[string][]byte{"p/p.go": []byte(large), "p/old.go": []byte(large)})) - productionNodes(base)
	paid := productionNodes(snapshot(map[string][]byte{"p/p.go": []byte(large)})) - productionNodes(base)
	ignored := productionNodes(snapshot(map[string][]byte{"p/p.go": []byte(small), "p/old.go": []byte(large),
		"p/p_test.go": []byte(large), "p/gen.go": []byte("// Code generated by x. DO NOT EDIT.\n\n" + large[len("package p\n"):]),
		"p/testdata/q.go": []byte(large)})) - productionNodes(base)
	if grown <= 0 || paid >= 0 || ignored != 0 {
		t.Fatalf("growth = %d, paid by a deletion = %d, in tests, generated and testdata files = %d", grown, paid, ignored)
	}
	for _, test := range []struct {
		budget plan.Budget
		growth int
		want   bool
	}{
		{plan.Budget{}, grown, false},
		{plan.Budget{}, paid, true},
		{plan.Budget{Reason: "a capability"}, grown, true},
		{plan.Budget{Reason: "a capability", Nodes: grown - 1}, grown, false},
		{plan.Budget{Reason: "must shrink", Nodes: paid}, paid, true},
		{plan.Budget{Reason: "must shrink", Nodes: paid - 1}, paid, false},
	} {
		if got := admits(test.budget, test.growth); got != test.want {
			t.Errorf("admits(%+v, %d) = %t, want %t", test.budget, test.growth, got, test.want)
		}
	}
}
