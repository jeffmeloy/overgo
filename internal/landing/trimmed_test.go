package landing

import (
	"crypto/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

// TestCommitCheckTestsHoldUnderTrimmedBuilds holds a commit's checks to building trimmed, which
// lets checkouts share the build cache: the things that failed when it was tried.
//
// The fix stage, under an environment that asks for trimmed builds, fixes the files of the
// checkout it runs in. Two checkouts hold the same bytes. The first is fixed, and its fixed file
// is then overwritten with a marker. The fix of the second fixes the second's file and leaves the
// marker alone. A trimmed go fix would replay the first checkout's result, which is kept under
// the first checkout's file names: it would write the marker over and leave the second unfixed.
// Each run puts bytes of its own in the package, so no entry an earlier run left answers it.
//
// The tests that find the tree hold when their own binary is built trimmed: a trimmed build
// compiles a test at its import path, where no file is, so each finds the tree from the directory
// it runs in.
func TestCommitCheckTestsHoldUnderTrimmedBuilds(t *testing.T) {
	// Serial: it asks for trimmed builds in the process environment, which the go commands the fix
	// stage starts read.
	t.Setenv("GOFLAGS", strings.TrimSpace(os.Getenv("GOFLAGS")+" -trimpath"))
	ctx := t.Context()
	const legacy = "internal/p/legacy.go"
	loop := "package p\n\n// Has reports whether values holds value, in the run " + rand.Text() + ".\nfunc Has(values []string, value string) bool {\n" +
		"\tfor _, candidate := range values {\n\t\tif candidate == value {\n\t\t\treturn true\n\t\t}\n\t}\n\treturn false\n}\n"
	// checkout is a committed module holding the loop go fix rewrites, with a change to another
	// file of the same package, so the fix stage has a changed package.
	checkout := func() *sliceFixture {
		t.Helper()
		fixture := newRowFixture(t, map[string]string{"internal/p/p.go": "package p\n", legacy: loop})
		fixture.write("internal/p/p.go", "// Package p is the fixture.\npackage p\n")
		return fixture
	}
	first, second := checkout(), checkout()
	if err := first.candidate().fix(ctx); err != nil {
		t.Fatal(err)
	}
	if fixed := first.read(legacy); !strings.Contains(fixed, "slices.Contains") {
		t.Fatalf("the fix stage left the first checkout's loop as it was:\n%s", fixed)
	}
	const marker = "package p\n\n// The first checkout's file, as the fix of another checkout must leave it.\n"
	first.write(legacy, marker)
	if err := second.candidate().fix(ctx); err != nil {
		t.Fatal(err)
	}
	if fixed := second.read(legacy); !strings.Contains(fixed, "slices.Contains") {
		t.Errorf("the fix stage left the second checkout's loop as it was, under a trimmed build:\n%s", fixed)
	}
	if kept := first.read(legacy); kept != marker {
		t.Errorf("the fix of the second checkout wrote the first checkout's file:\n%s", kept)
	}

	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, trimmed := range [][]string{
		{"./internal/clioptions", "^TestCLIInferenceDependencyBoundary$"},
		{"./internal/plan", "^(TestColibriCampaign|TestSinglePlanAuthority|TestDoctrineParagraphsNameWhatTheyGovern)$"},
	} {
		if output, err := clioptions.CombinedOutputIn(root, nil, "go", "test", "-trimpath", "-count=1", "-run", trimmed[1], trimmed[0]); err != nil {
			t.Errorf("built trimmed, the tests %s of %s do not pass: %v\n%s", trimmed[1], trimmed[0], err, output)
		}
	}
}
