package landing

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/testcheck"
)

// TestTheGateAdmitsInlineNumbersAndPackageExports holds the rules stage to what bears on behaviour, quality or
// measurement (fewer-refusals/trim-rules): a change that writes a number inline and exports a function its own package
// alone calls passes it. The two rules that refused both were behind 24 of the 30 refusals that named their rule by
// 2026-11-11; the policy against an arbitrary limit is held where it matters by its own checks, and a declaration
// nothing reaches by the reach check.
func TestTheGateAdmitsInlineNumbersAndPackageExports(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{"internal/p/p.go": emptyPackage})
	fixture.write("internal/p/p.go", emptyPackage+"\n// Half is the fixture's export, which its package alone calls.\nfunc Half(n int) int { return n / 2 }\n\n// Quarter halves twice.\nfunc quarter(n int) int { return Half(Half(n)) }\n")
	testcheck.NoError(t, fixture.candidate().rules(t.Context()))
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
	base := testcheck.Must((gosyntax.Snapshot{}).Overlay(map[string][]byte{"internal/p/p.go": []byte("package p\n")}))(t)
	candidate := testcheck.Must(base.Overlay(map[string][]byte{"internal/p/p.go": []byte(lazy)}))(t)
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
	snapshot := func(source string) gosyntax.Snapshot { return packageSnapshot(t, source) }
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

// TestTheGateNamesItsSlowestTests holds a stage's measured line to what go test ran: the packages it ran, slowest
// first, and their slowest top-level tests beside them, a package replayed from its cache left out with its tests,
// and a subtest left to its parent, whose time holds it.
func TestTheGateNamesItsSlowestTests(t *testing.T) {
	t.Parallel()
	output := strings.Join([]string{
		`{"Action":"pass","Package":"overgo/internal/cached","Test":"TestReplayed","Elapsed":900}`,
		`{"Action":"output","Package":"overgo/internal/cached","Output":"ok  \tovergo/internal/cached\t(cached)\n"}`,
		`{"Action":"pass","Package":"overgo/internal/cached","Elapsed":0.002}`,
		`{"Action":"run","Package":"overgo/internal/fast","Test":"TestQuick"}`,
		`{"Action":"fail","Package":"overgo/internal/fast","Test":"TestQuick","Elapsed":1}`,
		`{"Action":"fail","Package":"overgo/internal/fast","Elapsed":2}`,
		`{"Action":"pass","Package":"overgo/internal/slow","Test":"TestWhole/part","Elapsed":240}`,
		`{"Action":"pass","Package":"overgo/internal/slow","Test":"TestWhole","Elapsed":250}`,
		`{"Action":"pass","Package":"overgo/internal/slow","Test":"TestSmall","Elapsed":40}`,
		`{"Action":"pass","Package":"overgo/internal/slow","Elapsed":300}`,
	}, "\n")
	want := "ran 2 packages: internal/slow 300s, internal/fast 2s; slowest tests: internal/slow TestWhole 250s, internal/slow TestSmall 40s, internal/fast TestQuick 1s"
	if got := ranPackages(output); got != want {
		t.Fatalf("ranPackages = %q, want %q", got, want)
	}
}

// TestWrittenCodeIsCountedByKind holds what a landing counts as written Go:
// production growth is the nodes its production files gain, deletions pay for
// additions, a test file's nodes are test code and no part of it, and generated
// and testdata files are neither.
func TestWrittenCodeIsCountedByKind(t *testing.T) {
	t.Parallel()
	const small = "package p\n\nfunc A() int { return 1 }\n"
	const large = "package p\n\nfunc A() int { return 1 }\n\nfunc B() int { return A() + A() }\n"
	snapshot := func(files map[string][]byte) gosyntax.Snapshot {
		t.Helper()
		out := testcheck.Must(gosyntax.Snapshot{}.Overlay(files))(t)
		return out
	}
	base := snapshot(map[string][]byte{"p/p.go": []byte(small), "p/old.go": []byte(large)})
	with := snapshot(map[string][]byte{"p/p.go": []byte(small), "p/old.go": []byte(large),
		"p/p_test.go": []byte(large), "p/gen.go": []byte("// Code generated by x. DO NOT EDIT.\n\n" + large[len("package p\n"):]),
		"p/testdata/q.go": []byte(large), "p/testdata/q_test.go": []byte(large)})
	grown := writtenNodes(snapshot(map[string][]byte{"p/p.go": []byte(large), "p/old.go": []byte(large)}), false) - writtenNodes(base, false)
	paid := writtenNodes(snapshot(map[string][]byte{"p/p.go": []byte(large)}), false) - writtenNodes(base, false)
	ignored := writtenNodes(with, false) - writtenNodes(base, false)
	if grown <= 0 || paid >= 0 || ignored != 0 {
		t.Fatalf("growth = %d, paid by a deletion = %d, in tests, generated and testdata files = %d", grown, paid, ignored)
	}
	if tests, one := writtenNodes(with, true)-writtenNodes(base, true), writtenNodes(snapshot(map[string][]byte{"p/p.go": []byte(large)}), false); tests != one {
		t.Fatalf("test code grew by %d nodes, want the %d of the one test file outside testdata", tests, one)
	}
}

// TestLandingSaysWhatAChangeAddsOfEachKind holds the rules stage to the
// figures it writes into a landing's message beside the production growth it
// bounds: test code, record schemas, device kernel source with its entry
// points, and fixtures. A kernel file named as generated is output and is not
// counted, and a media type is not a schema.
func TestLandingSaysWhatAChangeAddsOfEachKind(t *testing.T) {
	t.Parallel()
	fixture := newRowFixture(t, map[string]string{
		"internal/p/p.go": "package p\n", "internal/p/testdata/old.json": "{}\n",
		"kernels/cuda/ops.cu": "__global__ void one() {}\n", "fixtures/held.bin": "a",
	})
	fixture.write("internal/p/p.go", "package p\n\nconst recordSchema = \"overgo.fixture-record.v1\"\n\nconst mediaType = \"application/vnd.overgo.fixture+json\"\n")
	fixture.write("internal/p/p_test.go", "package p\n\nconst again = recordSchema\n")
	fixture.write("internal/p/testdata/new.json", "{\"a\": 1}\n")
	fixture.write("fixtures/root.bin", "ab")
	testcheck.NoError(t, os.Remove(filepath.Join(fixture.repo, "internal", "p", "testdata", "old.json")))
	fixture.write("kernels/cuda/ops.cu", "__global__ void one() {}\n__global__ void two() {}\n__global__ void three() {}\n")
	fixture.write("kernels/cuda/tables_generated.cuh", "// Tables the __global__ kernels read.\nstatic const float table[] = {\n0,\n};\n")
	l := fixture.candidate()
	candidate, base, err := l.changedSource(t.Context())
	testcheck.NoError(t, err)
	production, tests := writtenNodes(candidate, false)-writtenNodes(base, false), writtenNodes(candidate, true)-writtenNodes(base, true)
	if production <= 0 || tests <= 0 {
		t.Fatalf("the change adds %d production and %d test nodes, want some of each", production, tests)
	}
	l.terms.Items[0].Budget.Nodes = production
	testcheck.NoError(t, l.rules(t.Context()))
	want := fmt.Sprintf("production %+d nodes of the %d its row may add, test code %+d nodes, record schemas +1, CUDA source +2 lines and +2 entry points, fixtures +1 files and +8 bytes", production, production, tests)
	if l.details["rules"] != want {
		t.Fatalf("the rules stage says %q, want %q", l.details["rules"], want)
	}
}
