package landing

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestPreflightRunsTheCheapStagesOnly holds the preflight to the stages that
// take seconds and refuse what the change itself got wrong, and the full
// landing's long stages out of it: every package's tests, the lanes and the
// scorecard slice run only when the change lands.
func TestPreflightRunsTheCheapStagesOnly(t *testing.T) {
	t.Parallel()
	var cheap, long []string
	for _, check := range (&landing{}).checks() {
		if check.cheap {
			cheap = append(cheap, check.name)
		} else {
			long = append(long, check.name)
		}
	}
	if want := []string{"format", "fix", "vet", "rules", "form", "verify", "invariants"}; !slices.Equal(cheap, want) {
		t.Errorf("the preflight runs %v, want %v", cheap, want)
	}
	if want := []string{"test", "lanes", "scorecard"}; !slices.Equal(long, want) {
		t.Errorf("the stages it leaves to the landing are %v, want %v", long, want)
	}
}

// TestALandingRunsEachSuiteOnce holds a landing to running each package's suite in one stage, the step's check before
// any (SQA-516): its invariants stage runs the whole-tree tests alone and its test stage every package's suite, the
// invariant packages' among them, where both stages had run those, internal/landing 252 s and then 292 s at ff47d11d6.
// A preflight, which has no test stage, runs the invariant suites in its invariants stage.
func TestALandingRunsEachSuiteOnce(t *testing.T) {
	t.Parallel()
	fixture := newSliceFixture(t, map[string]string{
		"internal/plan/plan.go":                   "// Package plan is a fixture.\npackage plan\n",
		"internal/plan/plan_test.go":              "package plan\n\nimport \"testing\"\n\nfunc TestHeld(t *testing.T) {}\n",
		"internal/landing/landing.go":             "// Package landing is a fixture.\npackage landing\n",
		"internal/landing/tree_wholetree_test.go": "//go:build wholetree\n\npackage landing\n\nimport \"testing\"\n\nfunc TestWhole(t *testing.T) {}\n",
	})
	held, whole := plan.PassedTest("fixture/internal/plan", "TestHeld"), plan.PassedTest("fixture/internal/landing", "TestWhole")
	for _, preflight := range []bool{false, true} {
		l := &landing{Options: Options{Repo: fixture.repo, Preflight: preflight}, details: map[string]string{}}
		testcheck.NoError(t, l.invariants(t.Context()))
		if !preflight {
			testcheck.NoError(t, l.test(t.Context()))
		}
		ran := map[string]int{}
		for _, passed := range l.passed {
			ran[passed]++
		}
		if ran[held] != 1 || ran[whole] != 1 {
			t.Errorf("a gate (preflight %t) ran the invariant suite %d times and the whole-tree test %d times; want each once", preflight, ran[held], ran[whole])
		}
	}
	var stages []string
	for _, check := range (&landing{}).checks() {
		stages = append(stages, check.name)
	}
	if verify := slices.Index(stages, "verify"); verify > slices.Index(stages, "invariants") || verify > slices.Index(stages, "test") {
		t.Errorf("a landing orders its stages %v; want the step's check before every suite", stages)
	}
}

// TestInvariantPackagesExist holds the invariant stage to packages the tree
// has: a renamed one would let the stage pass over nothing.
func TestInvariantPackagesExist(t *testing.T) {
	t.Parallel()
	for _, pattern := range invariantPackages {
		if info, err := os.Stat(filepath.Join("..", "..", filepath.FromSlash(pattern))); err != nil || !info.IsDir() {
			t.Errorf("invariant package %s is not a directory of the tree: %v", pattern, err)
		}
	}
}

// TestGeneratorsAreThoseTheRepositoryHas holds the derived-file stage to the
// commands a repository has: the API manifest and the compatibility claims are
// regenerated and rebound when their commands exist, and a tree without them,
// a fixture's, has nothing to derive.
func TestGeneratorsAreThoseTheRepositoryHas(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		commands []string
		want     []string
	}{
		{nil, nil},
		{[]string{"api-manifest"}, []string{"./cmd/api-manifest -update"}},
		{[]string{"api-manifest", "validate"}, []string{"./cmd/api-manifest -update", "./cmd/validate -claims -write"}},
	} {
		root := t.TempDir()
		for _, command := range test.commands {
			testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "cmd", command), 0o755))
		}
		var got []string
		for _, generator := range (&landing{Options: Options{Repo: root}}).generators() {
			got = append(got, strings.Join(generator.arguments[1:], " "))
		}
		if !slices.Equal(got, test.want) {
			t.Errorf("commands %v derive with %v, want %v", test.commands, got, test.want)
		}
	}
}
