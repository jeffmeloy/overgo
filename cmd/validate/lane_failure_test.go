package main

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds runs a lane over a module of three packages, in the
// order the models lane runs its own: the first's test fails, the second's lane tests took an hour in the runs before
// and its test would wait past the lane's deadline, and the third's took no wall worth recording. The failing test is
// reported with its output and added to the run record the moment it fails; the long package is skipped, never run,
// since a failed lane runs no cell; the short one still runs, and its test inherits no run record. The lane's
// verdict names both.
func TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module lanefixture\n\ngo 1.26\n")
	write("first/first_test.go", "package first\n\nimport \"testing\"\n\nfunc TestFails(t *testing.T) { t.Fatal(\"the answer is 5, want 4\") }\n")
	write("long/long_test.go", "package long\n\nimport \"testing\"\n\nfunc TestWaits(t *testing.T) { <-t.Context().Done() }\n")
	write("short/short_test.go", "package short\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\n"+
		"func TestRuns(t *testing.T) {\n\tif os.Getenv(\""+clioptions.MeasurementRunEnvironment+"\") != \"\" {\n\t\tt.Fatal(\"the lane's test was given the run record\")\n\t}\n}\n")
	if err := processlock.EnsureStateDirectory(root); err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(filepath.Join(root, processlock.StateDirectory, laneWallsFile), laneWalls{"lanefixture/long TestWaits": {3600}}, clioptions.PrivateFileMode); err != nil {
		t.Fatal(err)
	}
	record := filepath.Join(t.TempDir(), "measurement_run.json")
	started := time.Now()
	running := plan.MeasurementRun{Commit: "measured", Process: os.Getpid(), Started: started, Phases: []plan.MeasurementPhase{
		{Name: "preflight", Started: started, Ended: started, Passed: true}, {Name: "lanes", Started: started},
	}}
	if err := jsonfile.Write(record, running, clioptions.PrivateFileMode); err != nil {
		t.Fatal(err)
	}
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	var output bytes.Buffer
	err := acquire(t.Context(), &output, root, "go", []string{"test", "-p=1", "-timeout=10m", "./first", "./long", "./short"})
	if err == nil || !strings.Contains(err.Error(), "lanefixture/first TestFails failed") || !strings.Contains(err.Error(), "./long skipped after the failure") {
		t.Fatalf("the lane's verdict = %v, want the failed test and the skipped package named:\n%s", err, output.String())
	}
	reported := output.String()
	if !strings.Contains(reported, "validate -run: lanefixture/first TestFails failed:") || !strings.Contains(reported, "the answer is 5, want 4") {
		t.Fatalf("the failing test was not reported with its output:\n%s", reported)
	}
	if strings.Contains(reported, "lanefixture/long\t") || !strings.Contains(reported, "pass\tlanefixture/short") {
		t.Fatalf("the long package ran, or the short one did not pass, after the failure:\n%s", reported)
	}
	var kept plan.MeasurementRun
	if err := jsonfile.Decode(record, &kept); err != nil {
		t.Fatal(err)
	}
	if lanes := kept.Phases[1]; !slices.Equal(lanes.Failed, []string{"lanefixture/first TestFails"}) || len(kept.Phases[0].Failed) != 0 {
		t.Fatalf("the run record's phases = %+v, want the failed test in the running lanes phase alone", kept.Phases)
	}
	if report := kept.Report(); !strings.Contains(report, "lanes running since") || !strings.Contains(report, "failed: lanefixture/first TestFails") {
		t.Fatalf("the run reads %q", report)
	}
}
