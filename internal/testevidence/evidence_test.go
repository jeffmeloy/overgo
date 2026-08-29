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
	report, err := GoTestJSONReport(classified)
	if err != nil {
		t.Fatal(err)
	}
	if err := RequireComplete(report); err == nil {
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
	if err := RequireComplete(report); err == nil {
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

func TestVerifyGoTestEvidenceWithoutRun(t *testing.T) {
	passing := "{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"x\"}\n"
	if err := VerifyGoTestEvidence("go test ./x -count=1", passing); err != nil {
		t.Fatal(err)
	}
	if err := VerifyGoTestEvidence("go test ./x -count=1", "{\"Action\":\"pass\",\"Package\":\"x\"}\n"); err == nil {
		t.Fatal("package-only output passed without an executed test")
	}
	skipped := "{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"x\"}\n"
	if err := VerifyGoTestEvidence("go test ./x -count=1", skipped); err == nil {
		t.Fatal("broad acceptance credited a skipped test")
	}
}

func TestVerifyGoTestEvidenceClassifiesExplicitShortExclusions(t *testing.T) {
	passing := "{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"x\"}\n"
	classified := fmt.Sprintf(
		"{\"Action\":\"output\",\"Package\":\"x\",\"Test\":\"TestIntegration\",\"Output\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestIntegration\"}\n",
		ShortIntegrationSkip+"\n",
	)
	if err := VerifyGoTestEvidence("go test -race -short ./x", passing+classified); err != nil {
		t.Fatal(err)
	}
	if err := VerifyGoTestEvidence("go test -race ./x", passing+classified); err == nil {
		t.Fatal("non-short verifier accepted a short exclusion")
	}
	if err := VerifyGoTestEvidence("go test -race -short ./x", passing+"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestMystery\"}\n"); err == nil {
		t.Fatal("short verifier accepted an unclassified skip")
	}
}

func TestVerifyGoTestEvidenceMixedCommand(t *testing.T) {
	passing := "{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"x\"}\n"
	command := "go test ./x && go run ./cmd/device-lane"
	if err := VerifyGoTestEvidence(command, passing+"device lane green\n"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyGoTestEvidence(command, passing+"{malformed event\n"); err == nil {
		t.Fatal("mixed verifier accepted malformed JSON event")
	}
	if err := VerifyGoTestEvidence(command, passing+"DEVICE UNAVAILABLE\n"); err == nil {
		t.Fatal("mixed verifier accepted unavailable auxiliary evidence")
	}
	if err := VerifyGoTestEvidence("go test ./x", passing+"unexpected output\n"); err == nil {
		t.Fatal("test-only verifier accepted auxiliary output")
	}
}

func TestFailureSummary(t *testing.T) {
	evidence := "{\"Action\":\"fail\",\"Package\":\"x\",\"Test\":\"TestBroken\"}\n" +
		"{\"Action\":\"fail\",\"Package\":\"x\"}\n" +
		"device diagnostics\n"
	if got := FailureSummary(evidence); got != "x: TestBroken, x" {
		t.Fatalf("failure summary = %q", got)
	}
}
