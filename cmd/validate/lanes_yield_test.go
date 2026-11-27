package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestALanesRunYieldsAtAPackageToANewerClose holds a run of the lanes a runner starts to yielding, as a lane turns to
// another package, to a newer commit named due (lanes-first/yield): a lane of two fixture packages whose newer commit
// is named due once the first has run runs the first and not the second, keeps the first's test's pass as it ended
// and nothing of the second (lanes-record), and ends without a failure; and a runner's yield reads the due file against
// the run's own commit. moss-attention/do landed at 00:46 on 2026-11-19 while a run at an older commit ran on for 11
// hours.
func TestALanesRunYieldsAtAPackageToANewerClose(t *testing.T) {
	t.Parallel()
	root, head := fixtureRepository(t)
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644))
	for _, pkg := range []string{"first", "second"} {
		testcheck.NoError(t, os.MkdirAll(filepath.Join(root, pkg), 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, pkg, pkg+"_test.go"), []byte("package "+pkg+"\n\nimport \"testing\"\n\nfunc TestRuns(t *testing.T) {}\n"), 0o644))
	}
	asked := 0
	lanes := Plan{
		Suites: []SelectedSuite{{Lane: "models", Arguments: []string{"test", "-count=1", "./first", "./second"},
			Each: [][]string{{"test", "-count=1", "./first"}, {"test", "-count=1", "./second"}}}},
		LanesKept: t.TempDir(),
		Yields: func() (string, bool) {
			asked++
			return "newer", asked > 1
		},
	}
	var out bytes.Buffer
	err := executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes)
	results := plan.ReadTestResults(lanes.LanesKept)
	if err != nil || !strings.Contains(out.String(), "yields to newer") || strings.Contains(out.String(), "./second\n") {
		t.Errorf("a run told after its first package that a newer commit is due ended %v; want it yielded with no failure, its second package not run\n%s", err, out.String())
	}
	if first := results["fixture/first TestRuns"]; first.Commit != head || first.Failed || len(results) != 1 {
		t.Errorf("a run that yielded after its first package kept %+v; want the first package's test's pass alone, at %s", results, head)
	}

	due := filepath.Join(t.TempDir(), "lanes_due.json")
	yield := yieldTo(t.Context(), root, due)
	testcheck.NoError(t, os.WriteFile(due, []byte(`{"commit":"`+head+`","rows":["fixture"]}`), 0o644))
	if to, yields := yield.yields(); yields {
		t.Errorf("a run yielded to %q, its own commit named due; want it to run on", to)
	}
	testcheck.NoError(t, os.WriteFile(due, []byte(`{"commit":"a-newer-close","rows":["fixture"]}`), 0o644))
	if to, yields := yield.yields(); !yields || to != "a-newer-close" {
		t.Errorf("a run with a newer commit named due yielded %t to %q; want it yielded to that commit", yields, to)
	}
}
