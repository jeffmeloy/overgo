package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
)

// TestTheRunRecordHoldsEachPackagesWallAndPeak runs a lane over a module of two packages, one at a time as the long
// lanes run theirs: the first allocates little and the second commits a quarter gigabyte. Each package is added to
// the running phase of the run record as it ends, with the wall go test measured and the most memory one process of
// the lane had committed by then, so the package that raised the lane's peak is the one whose peak exceeds the one
// before. Windows' job counter holds the peak; elsewhere it reads zero.
func TestTheRunRecordHoldsEachPackagesWallAndPeak(t *testing.T) {
	root := t.TempDir()
	const committed = 256 << 20
	writeFixture := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeFixture("go.mod", "module lanefixture\n\ngo 1.26\n")
	writeFixture("small/small_test.go", "package small\n\nimport \"testing\"\n\nfunc TestRuns(t *testing.T) {}\n")
	writeFixture("large/large_test.go", "package large\n\nimport \"testing\"\n\nfunc TestCommits(t *testing.T) {\n\tdata := make([]byte, 256<<20)\n\tfor index := range data {\n\t\tdata[index] = 1\n\t}\n\tif data[len(data)-1] != 1 {\n\t\tt.Fatal(\"unwritten\")\n\t}\n}\n")
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
	if err := acquire(t.Context(), &output, root, "go", []string{"test", "-p=1", "-timeout=10m", "./small", "./large"}); err != nil {
		t.Fatalf("the lane failed: %v\n%s", err, output.String())
	}
	var kept plan.MeasurementRun
	if err := jsonfile.Decode(record, &kept); err != nil {
		t.Fatal(err)
	}
	ran := kept.Phases[1].Packages
	if len(kept.Phases[0].Packages) != 0 || len(ran) != 2 || ran[0].Package != "lanefixture/small" || ran[1].Package != "lanefixture/large" {
		t.Fatalf("the run record's phases hold packages %+v and %+v, want the lane's two in the running lanes phase", kept.Phases[0].Packages, ran)
	}
	if ran[0].Wall <= 0 || ran[1].Wall <= 0 {
		t.Fatalf("the packages' walls are %v and %v, want go test's", ran[0].Wall, ran[1].Wall)
	}
	if runtime.GOOS == "windows" && (ran[1].PeakBytes < committed || ran[1].PeakBytes <= ran[0].PeakBytes) {
		t.Fatalf("the lane's peak is %d bytes after the small package and %d after the large, want the large to raise it past %d", ran[0].PeakBytes, ran[1].PeakBytes, committed)
	}
}
