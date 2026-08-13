package testevidence

import "testing"

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

func TestVerifyOutput(t *testing.T) {
	tests := []struct {
		name, command, output string
		wantErr               bool
	}{
		{name: "pass", command: "go test ./x -v", output: "--- PASS: TestX (0.00s)\nPASS\n"},
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
