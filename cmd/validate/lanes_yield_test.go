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
// is named due once the first has run runs the first and not the second, keeps the first's pass and not the lane's,
// and ends yielded, not failed; and a runner's yield reads the due file against the run's own commit, writing its
// record beside the file. moss-attention/do landed at 00:46 on 2026-11-19 while a run at an older commit ran on for 11
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
	record := plan.ReadLaneRuns(lanes.LanesKept)["models"]
	if err == nil || !strings.Contains(err.Error(), errYielded.Error()) || strings.Contains(out.String(), "./second\n") {
		t.Errorf("a run told after its first package that a newer commit is due ended %v; want it yielded, its second package not run\n%s", err, out.String())
	}
	if record.Last.Passed || record.Packages["./first"].Commit != head || record.Packages["./second"].Commit != "" {
		t.Errorf("a run that yielded after its first package kept %+v; want the first package's pass alone, the lane not passed", record)
	}

	due := filepath.Join(t.TempDir(), "lanes_due.json")
	yields := yieldTo(t.Context(), root, due)
	testcheck.NoError(t, os.WriteFile(due, []byte(`{"commit":"`+head+`","rows":["fixture"]}`), 0o644))
	if to, yield := yields(); yield {
		t.Errorf("a run yielded to %q, its own commit named due; want it to run on", to)
	}
	testcheck.NoError(t, os.WriteFile(due, []byte(`{"commit":"a-newer-close","rows":["fixture"]}`), 0o644))
	to, yield := yields()
	data, readErr := os.ReadFile(filepath.Join(filepath.Dir(due), plan.LanesYieldedFile))
	if !yield || to != "a-newer-close" || readErr != nil || !strings.Contains(string(data), head) {
		t.Errorf("a run with a newer commit named due yielded %t to %q, its record %q (%v); want it yielded to that commit, its record naming its own %s",
			yield, to, data, readErr, head)
	}
}
