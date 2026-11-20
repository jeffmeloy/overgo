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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds runs a lane over a module of three packages, in the
// order the models lane runs its own: the first's test fails, the second's lane tests took an hour in the runs before,
// and the third's took no wall worth recording. The failing test is reported with its output and added to the run
// record the moment it fails; the long package and the short one still run to their ends, so the run names every break
// there is (a failed lane's long packages were skipped by their walls, no-clocks/own-budgets), and the short one's
// test inherits no run record. The lane's verdict names the failure and nothing skipped.
func TestALaneTestThatFailsIsRecordedBeforeTheNextPackageEnds(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) { testutil.WriteTextFile(t, root, name, content, 0o600) }
	write("go.mod", "module lanefixture\n\ngo 1.26\n")
	write("first/first_test.go", "package first\n\nimport \"testing\"\n\nfunc TestFails(t *testing.T) { t.Fatal(\"the answer is 5, want 4\") }\n")
	write("long/long_test.go", "package long\n\nimport \"testing\"\n\nfunc TestRuns(t *testing.T) {}\n")
	write("short/short_test.go", "package short\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\n"+
		"func TestRuns(t *testing.T) {\n\tif os.Getenv(\""+clioptions.MeasurementRunEnvironment+"\") != \"\" {\n\t\tt.Fatal(\"the lane's test was given the run record\")\n\t}\n}\n")
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	record := filepath.Join(t.TempDir(), "measurement_run.json")
	started := time.Now()
	running := plan.MeasurementRun{Commit: "measured", Process: os.Getpid(), Started: started, Phases: []plan.MeasurementPhase{
		{Name: "preflight", Started: started, Ended: started, Passed: true}, {Name: "lanes", Started: started},
	}}
	testcheck.NoError(t, jsonfile.Write(record, running, clioptions.PrivateFileMode))
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	var output bytes.Buffer
	_, err := acquire(t.Context(), &output, root, "go", []string{"test", "-count=1", "-p=1", "-timeout=0", "./first", "./long", "./short"})
	if err == nil || !strings.Contains(err.Error(), "lanefixture/first: TestFails failed") || strings.Contains(err.Error(), "skipped") {
		t.Fatalf("the lane's verdict = %v, want the failed test named and nothing skipped:\n%s", err, output.String())
	}
	reported := output.String()
	if !strings.Contains(reported, "validate -run: lanefixture/first TestFails failed:") || !strings.Contains(reported, "the answer is 5, want 4") {
		t.Fatalf("the failing test was not reported with its output:\n%s", reported)
	}
	if !strings.Contains(reported, "pass\tlanefixture/long") || !strings.Contains(reported, "pass\tlanefixture/short") {
		t.Fatalf("the long package or the short one did not run to its end after the failure:\n%s", reported)
	}
	var kept plan.MeasurementRun
	testcheck.NoError(t, jsonfile.Decode(record, &kept))
	if lanes := kept.Phases[1]; !slices.Equal(lanes.Failed, []string{"lanefixture/first TestFails"}) || len(kept.Phases[0].Failed) != 0 {
		t.Fatalf("the run record's phases = %+v, want the failed test in the running lanes phase alone", kept.Phases)
	}
	if report := kept.Report(); !strings.Contains(report, "lanes running since") || !strings.Contains(report, "failed: lanefixture/first TestFails") {
		t.Fatalf("the run reads %q", report)
	}
}

// TestAnEndedTestIsLoggedAsItEnds runs a lane over a package whose test ends its go test process from outside, as a
// tree's bound ends one, so go test reports nothing of it (lanes-ended): the test is reported as failed the moment its
// process ends, with how it ended and its largest process's memory, and recorded in the run, where ten tests the run
// at f46760944 ended at its bound reached the lanes log only in the run's summary, 8h32m after the run began. A
// package that ends is reported with its largest process's memory.
func TestAnEndedTestIsLoggedAsItEnds(t *testing.T) {
	root := t.TempDir()
	write := func(name, content string) { testutil.WriteTextFile(t, root, name, content, 0o600) }
	write("go.mod", "module lanefixture\n\ngo 1.26\n")
	write("ran/ran_test.go", "package ran\n\nimport \"testing\"\n\nfunc TestRuns(t *testing.T) {}\n")
	// The test waits for go test to have said it runs, then ends go test, its parent, and with it the run.
	write("ended/ended_test.go", "package ended\n\nimport (\n\t\"os\"\n\t\"testing\"\n\t\"time\"\n)\n\n"+
		"func TestEndsItsRun(t *testing.T) {\n\ttime.Sleep(2 * time.Second)\n\tparent, err := os.FindProcess(os.Getppid())\n"+
		"\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n\t_ = parent.Kill()\n\ttime.Sleep(time.Minute)\n}\n")
	testcheck.NoError(t, processlock.EnsureStateDirectory(root))
	record := filepath.Join(t.TempDir(), "measurement_run.json")
	started := time.Now()
	testcheck.NoError(t, jsonfile.Write(record, plan.MeasurementRun{Commit: "measured", Process: os.Getpid(), Started: started,
		Phases: []plan.MeasurementPhase{{Name: "lanes", Started: started}}}, clioptions.PrivateFileMode))
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	var output bytes.Buffer
	_, err := acquire(t.Context(), &output, root, "go", []string{"test", "-count=1", "-p=1", "-timeout=0", "./ran", "./ended"})
	reported := output.String()
	if err == nil || !strings.Contains(reported, "fail\tlanefixture/ended\tTestEndsItsRun\t") ||
		!strings.Contains(reported, "validate -run: lanefixture/ended TestEndsItsRun ended without its result, its largest process at ") {
		t.Fatalf("a test whose process ended without its result ended the lane with %v, said as:\n%s", err, reported)
	}
	if !strings.Contains(reported, "pass\tlanefixture/ran\t") || !strings.Contains(reported, " GiB\n") {
		t.Fatalf("the package that ran was not said with its largest process's memory:\n%s", reported)
	}
	var kept plan.MeasurementRun
	testcheck.NoError(t, jsonfile.Decode(record, &kept))
	if !slices.Equal(kept.Phases[0].Failed, []string{"lanefixture/ended TestEndsItsRun"}) {
		t.Fatalf("the run record's lanes phase = %+v, want the ended test failed", kept.Phases[0])
	}
}
