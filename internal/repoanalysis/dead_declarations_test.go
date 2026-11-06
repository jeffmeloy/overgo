package repoanalysis

import (
	"path"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// reachedByNothing lists the keys of the functions and methods of the module at root that no command and no test reaches.
func reachedByNothing(t *testing.T, root string, outside map[string]bool) []string {
	t.Helper()
	reach := testcheck.Must(reachTypes(t.Context(), root, outside, true))(t)
	var nothing []string
	for _, function := range reach.functions {
		if !reach.entered[function.at] {
			nothing = append(nothing, function.key)
		}
	}
	slices.Sort(nothing)
	return slices.Compact(nothing)
}

// TestNoDeclarationIsReachedByNothing holds the census's dead class to what nothing reaches: a function or method no
// command and no test reaches is kept as dead, or as a prototype of a model family, which stays, and one kept as dead
// is reached by nothing. A test reaches what it names, and what test support it calls names, behind any build tag the
// tests build behind. The dead outside the inference surface are deleted, and those inside wait for surface-folds,
// which moves the surface's keys once.
func TestNoDeclarationIsReachedByNothing(t *testing.T) {
	t.Parallel()
	fixture := fixtureModule(t, map[string]string{
		"cmd/tool/main.go":            "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { lib.Use() }\n",
		"internal/lib/lib.go":         "package lib\n\nfunc Use() {}\n\nfunc Tested() {}\n\nfunc Supported() {}\n\nfunc Idle() {}\n",
		"internal/lib/lib_test.go":    "package lib\n\nimport \"testing\"\n\nfunc TestTested(t *testing.T) { Tested() }\n",
		"internal/libtest/libtest.go": "package libtest\n\nimport \"overgo/internal/lib\"\n\nfunc Fixture() { lib.Supported() }\n",
		"internal/other/other.go":     "package other\n",
		"internal/other/other_test.go": "package other\n\nimport (\n\t\"testing\"\n\n\t\"overgo/internal/libtest\"\n)\n\n" +
			"func TestOther(t *testing.T) { libtest.Fixture() }\n",
	})
	if got := reachedByNothing(t, fixture, nil); !slices.Equal(got, []string{"internal/lib/lib.go Idle"}) {
		t.Errorf("a function a test calls, and one test support calls for a test, are reached, and one nothing calls is not: reached by nothing = %q, want only Idle", got)
	}

	outside := testcheck.Must(outsideMethods(t.Context(), checkoutRoot))(t)
	nothing := reachedByNothing(t, checkoutRoot, outside)
	kept, classOf, err := sortedEntries(keptList(t, unreachedList))
	if err != nil {
		t.Fatalf("%s: %v", unreachedList, err)
	}
	var dead []string
	for _, entry := range kept {
		if classOf[entry] == "dead" {
			dead = append(dead, entry)
		}
	}
	missing, stale := listDrift(nothing, dead)
	for _, key := range missing {
		if classOf[key] != "prototype" {
			t.Errorf("no command and no test reaches %s, so delete it, or keep it as dead inside the inference surface until surface-folds", key)
		}
	}
	for _, key := range stale {
		t.Errorf("%s keeps %s as dead, and a command or a test reaches it, so sort it again", unreachedList, key)
	}
	inside := surfaceDirectories(t)
	for _, entry := range dead {
		if file, _, _ := strings.Cut(entry, " "); !inside[path.Dir(file)] {
			t.Errorf("%s is dead outside the inference surface, so delete it, its line and its group's", entry)
		}
	}
}
