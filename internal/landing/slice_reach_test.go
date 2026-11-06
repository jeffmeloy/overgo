package landing

import (
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestATestOnlyChangeRunsNoScorecardSlice holds the scorecard slice to what its benchmark's binary builds from, so a
// change of tests alone measures no speed: in this checkout the benchmark's own test file, the test support only tests
// import and a production package's test data reach no slice, and the benchmark's own source and a production package
// its binary imports do. A fixture module pins the binary reading against a lane over the same package that runs its
// tests.
func TestATestOnlyChangeRunsNoScorecardSlice(t *testing.T) {
	t.Parallel()
	reaches := testcheck.Must(discover(testutil.RepoRoot(t), slice))(t)
	for _, change := range []struct {
		path    string
		reached bool
	}{
		{"cmd/benchmark/main_test.go", false},
		{"internal/testutil/fixture.go", false},
		{"internal/testcheck/testcheck.go", false},
		{"internal/inference/testdata/golden.json", false},
		{"cmd/benchmark/main.go", true},
		{"internal/inference/runner_generation.go", true},
	} {
		if got := reaches(slice, []string{change.path}).by != ""; got != change.reached {
			t.Errorf("a change to %s reached the scorecard slice = %t, want %t", change.path, got, change.reached)
		}
	}
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":                "module fixture\n\ngo 1.26\n",
		"tool/main.go":          "package main\n\nimport \"fixture/lib\"\n\nfunc main() { _ = lib.Value }\n",
		"tool/main_test.go":     "package main\n\nimport (\n\t\"testing\"\n\n\t\"fixture/support\"\n)\n\nfunc TestTool(t *testing.T) { _ = support.Value }\n",
		"lib/lib.go":            "package lib\n\nvar Value = 1\n",
		"lib/testdata/case.txt": "case\n",
		"support/support.go":    "package support\n\nvar Value = 2\n",
	})
	binary, tests := lane{packages: []string{"./tool"}, binary: true}, lane{packages: []string{"./tool"}}
	fixtureReaches := testcheck.Must(discover(repo, binary, tests))(t)
	for _, change := range []struct {
		path                 string
		binary, testsReached bool
	}{
		{"tool/main.go", true, true},
		{"lib/lib.go", true, true},
		{"tool/main_test.go", false, true},
		{"support/support.go", false, true},
		{"lib/testdata/case.txt", false, true},
	} {
		if got := fixtureReaches(binary, []string{change.path}).by != ""; got != change.binary {
			t.Errorf("a change to %s reached the binary's lane = %t, want %t", change.path, got, change.binary)
		}
		if got := fixtureReaches(tests, []string{change.path}).by != ""; got != change.testsReached {
			t.Errorf("a change to %s reached the tests' lane = %t, want %t", change.path, got, change.testsReached)
		}
	}
}
