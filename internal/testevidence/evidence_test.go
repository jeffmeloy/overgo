package testevidence

import (
	"fmt"
	"testing"
)

func TestGoTestJSONShort(t *testing.T) {
	classified := fmt.Sprintf(
		"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestX\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestX\"}\n",
		ShortIntegrationSkip+"\n",
	)
	if err := GoTestJSONShort(classified); err != nil {
		t.Fatal(err)
	}
	if err := GoTestJSON(classified); err == nil {
		t.Fatal("strict evidence accepted a classified skip")
	}
	unclassified := "{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestX\"}\n"
	if err := GoTestJSONShort(unclassified); err == nil {
		t.Fatal("short evidence accepted an unclassified skip")
	}
}

func TestGoTestJSONReportPreservesSkippedEvidence(t *testing.T) {
	out := "{\"Action\":\"skip\",\"Package\":\"overgo/example\",\"Test\":\"TestFixture\",\"Output\":\"UNAVAILABLE: fixture\\n\"}\n"
	report, err := GoTestJSONReport(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Skipped) != 1 || len(report.Unavailable) != 1 {
		t.Fatalf("report = %+v", report)
	}
	if err := GoTestJSON(out); err == nil {
		t.Fatal("strict evidence accepted a skipped fixture")
	}
}

func TestCIRequiredEvidence(t *testing.T) {
	out := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestPass\"}\n"+
			"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestSlow\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestSlow\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"x\"}\n",
		ShortIntegrationSkip+"\n",
	)
	report, err := GoTestJSONShortReport(out)
	if err != nil {
		t.Fatal(err)
	}
	if report.PassedTests != 1 || report.PassedPackages != 1 || len(report.ClassifiedSkipped) != 1 {
		t.Fatalf("required evidence counts are dishonest: %+v", report)
	}
	if len(report.Skipped) != 0 || len(report.Unavailable) != 0 {
		t.Fatalf("required evidence unexpectedly incomplete: %+v", report)
	}
	if err := GoTestJSONShort(out + "{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestMystery\"}\n"); err == nil {
		t.Fatal("CI evidence accepted an unclassified skip")
	}
}

func TestVerifyOutput(t *testing.T) {
	tests := []struct {
		name, command, output string
		wantErr               bool
	}{
		{name: "pass", command: "go test ./x -v", output: "--- PASS: TestX (0.00s)\nPASS\n"},
		{name: "benchmark", command: "go test ./x -bench BenchmarkX", output: "BenchmarkX-8  1  120 ns/op\nPASS\n"},
		{name: "benchmark name only", command: "go test ./x -bench BenchmarkX", output: "BenchmarkX\nPASS\n", wantErr: true},
		{name: "skip", command: "go test ./x -v", output: "--- SKIP: TestX (0.00s)\nPASS\n", wantErr: true},
		{name: "quiet", command: "go test ./x", output: "ok\tx\t0.1s\n", wantErr: true},
		{name: "unavailable", command: "go test ./x -v", output: "UNAVAILABLE\n--- PASS: TestX\n", wantErr: true},
		{name: "non-go", command: "test -s x", output: ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := VerifyOutput(test.command, test.output) != nil; got != test.wantErr {
				t.Fatalf("error = %v, want %v", got, test.wantErr)
			}
		})
	}
}
