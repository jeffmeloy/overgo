package testevidence

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/runrecord"
	"overgo/internal/testskip"
)

func TestVerifyGoTestEvidenceSubtests(t *testing.T) {
	fixture := filepath.Join(t.TempDir(), "selector_test.go")
	const source = `package selector
import "testing"
func TestProbe(t *testing.T) {
 t.Run("child", func(t *testing.T) { t.Run("leaf", func(t *testing.T) {}) })
 t.Run("space name", func(t *testing.T) {})
 t.Run("skip", func(t *testing.T) { t.Skip("fixture skip") })
 t.Run("fail", func(t *testing.T) { t.Fatal("fixture failure") })
}
func TestOther(t *testing.T) {}
func TestSkippedParent(t *testing.T) { t.Run("child", func(t *testing.T) { t.Skip("fixture skip") }) }
`
	if err := os.WriteFile(fixture, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	var passing string
	for _, tc := range []struct {
		pattern string
		accept  bool
	}{
		{`^TestProbe$/^child$`, true},
		{`^TestProbe$/^child$/^leaf$`, true},
		{`^TestOther$|^TestProbe$/^child$`, true},
		{`^TestProbe$/^[c/|]hild$`, true},
		{`^(TestOther|TestProbe)$/^child$`, true},
		{`^TestProbe$/^space name$`, true},
		{`^TestProbe\/child$`, false},
		{`^(TestProbe/child)$`, false},
		{`^TestProbe$/^missing$`, false},
		{`^TestProbe$/^skip$`, false},
		{`^TestProbe$/^fail$`, false},
		{`^TestProbe$`, false},
		{`^TestSkippedParent$`, false},
		{`^TestProbe$/[`, false},
		{`^TestProbe$/\`, false},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			out, runErr := exec.CommandContext(t.Context(), "go", "test", "-json", fixture, "-run", tc.pattern, "-count=1").CombinedOutput()
			command := "go test ./fixture -run '" + tc.pattern + "' -count=1"
			if err := VerifyGoTestEvidence(command, string(out)); (err == nil) != tc.accept {
				t.Errorf("evidence = %v, accept = %v; go test = %v\n%s", err, tc.accept, runErr, out)
			}
			if tc.accept && runErr != nil {
				t.Fatalf("native Go selector failed: %v\n%s", runErr, out)
			}
			if tc.pattern == `^TestProbe$/^child$` {
				passing = string(out)
			}
		})
	}
	const command = "go test ./fixture -run '^TestProbe$/^child$'"
	compound := command + " && " + command
	if err := VerifyGoTestEvidence(compound, passing+passing); err != nil {
		t.Fatalf("separate complete invocations rejected: %v", err)
	}
	other := strings.ReplaceAll(passing, "TestProbe", "TestAnother")
	guarded := "test -f fixture.go && " + command + " && " + strings.ReplaceAll(command, "TestProbe", "TestAnother")
	if err := VerifyGoTestEvidence(guarded, passing+"fixture present\n"+other); err != nil {
		t.Fatalf("guarded distinct selectors rejected: %v", err)
	}
	if err := VerifyGoTestEvidence(guarded, passing+passing); err == nil {
		t.Fatal("first selector hid a missing second selector")
	}
	if err := RepeatAgreementForCommand(compound, passing+other, other+passing); err == nil {
		t.Fatal("merged test names hid changed invocation verdicts")
	}
	otherPackage := strings.ReplaceAll(passing, "command-line-arguments", "example/other")
	if err := RepeatAgreementForCommand(command, passing+otherPackage, otherPackage+passing); err != nil {
		t.Fatalf("independent package ordering changed verdicts: %v", err)
	}
	for _, mutation := range []string{"package terminal", "child terminal", "child absent", "child skipped", "child failed", "unrelated skip", "package contradicted"} {
		t.Run(mutation, func(t *testing.T) {
			var out strings.Builder
			for line := range strings.SplitSeq(passing, "\n") {
				if line == "" {
					continue
				}
				var event goTestEvent
				if err := json.Unmarshal([]byte(line), &event); err != nil {
					t.Fatal(err)
				}
				child := strings.HasPrefix(event.Test, "TestProbe/child")
				if mutation == "package terminal" && event.Test == "" && event.Action == "pass" || mutation == "child terminal" && child && event.Action == "pass" || mutation == "child absent" && child {
					continue
				}
				if child && event.Action == "pass" {
					switch mutation {
					case "child skipped":
						event.Action = "skip"
					case "child failed":
						event.Action = "fail"
					}
				}
				if event.Test == "" && event.Action == "pass" {
					extra := event
					if mutation == "unrelated skip" {
						extra.Test, extra.Action = "TestOther", "skip"
					}
					if mutation == "package contradicted" {
						extra.Action = "fail"
					}
					if mutation == "unrelated skip" || mutation == "package contradicted" {
						if err := json.NewEncoder(&out).Encode(extra); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := json.NewEncoder(&out).Encode(event); err != nil {
					t.Fatal(err)
				}
			}
			if err := VerifyGoTestEvidence(command, out.String()); (err == nil) != (mutation == "unrelated skip") {
				t.Fatalf("mutated evidence = %v", err)
			}
			if err := VerifyGoTestEvidence(compound, passing+out.String()); (err == nil) != (mutation == "unrelated skip") {
				t.Fatalf("earlier pass hid later incomplete invocation: %v", err)
			}
			if err := RepeatAgreementForCommand(command, passing, out.String()); err == nil && mutation != "package terminal" && mutation != "package contradicted" {
				t.Fatal("changed verdicts received repeat credit")
			}
		})
	}
	if err := VerifyGoTestEvidence(command, ""); err == nil {
		t.Fatal("empty evidence passed")
	}
	for _, pattern := range []string{`^TestProbe$/[`, `^TestProbe$/\`} {
		if _, err := runrecord.GoTestTargets("go test ./fixture -run '"+pattern+"'", true); err == nil {
			t.Fatalf("malformed selector %q passed admission", pattern)
		}
	}
}

func TestGoTestJSONShort(t *testing.T) {
	classified := fmt.Sprintf(
		"{\"Action\":\"attr\",\"Package\":\"x\",\"Test\":\"TestX\",\"Key\":%q,\"Value\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestX\"}\n",
		testskip.Key, testskip.KindShort,
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
	if len(report.Skipped) != 1 {
		t.Fatalf("report = %+v", report)
	}
	if err := RequireComplete(report); err == nil {
		t.Fatal("strict evidence accepted a skipped fixture")
	}
}

func TestCIRequiredEvidence(t *testing.T) {
	out := fmt.Sprintf(
		"{\"Action\":\"pass\",\"Package\":\"x\",\"Test\":\"TestPass\"}\n"+
			"{\"Action\":\"attr\",\"Package\":\"x\",\"Test\":\"TestSlow\",\"Key\":%q,\"Value\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestSlow\"}\n"+
			"{\"Action\":\"pass\",\"Package\":\"x\"}\n",
		testskip.Key, testskip.KindShort,
	)
	report, err := GoTestJSONShortReport(out)
	if err != nil {
		t.Fatal(err)
	}
	if report.PassedTests != 1 || report.PassedPackages != 1 || len(report.ClassifiedSkipped) != 1 {
		t.Fatalf("required evidence counts are wrong: %+v", report)
	}
	if len(report.Skipped) != 0 {
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
		"{\"Action\":\"attr\",\"Package\":\"x\",\"Test\":\"TestIntegration\",\"Key\":%q,\"Value\":%q}\n"+
			"{\"Action\":\"skip\",\"Package\":\"x\",\"Test\":\"TestIntegration\"}\n",
		testskip.Key, testskip.KindShort,
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

// TestVerifyGoTestEvidencePairsSelectorsPerInvocation holds a chained
// verifier to being judged invocation by invocation: the targeted invocation
// answers for its packages' reports and the broad one for its own, an
// auxiliary command between them yields nothing, and each invocation must
// have produced a report that meets its own contract.
func TestVerifyGoTestEvidencePairsSelectorsPerInvocation(t *testing.T) {
	command := "OVERGO_DATA_ROOT=C:/root go test ./cmd/x -run '^TestOne$' -count=1 && go test ./internal/y -count=1 && go run ./cmd/lane -diagnostic"
	targeted := "{\"Action\":\"start\",\"Package\":\"overgo/cmd/x\"}\n" +
		"{\"Action\":\"run\",\"Package\":\"overgo/cmd/x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/cmd/x\",\"Test\":\"TestOne\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/cmd/x\"}\n"
	broad := "{\"Action\":\"start\",\"Package\":\"overgo/internal/y\"}\n" +
		"{\"Action\":\"run\",\"Package\":\"overgo/internal/y\",\"Test\":\"TestTwo\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/internal/y\",\"Test\":\"TestTwo\"}\n" +
		"{\"Action\":\"pass\",\"Package\":\"overgo/internal/y\"}\n"
	if err := VerifyGoTestEvidence(command, targeted+broad+"lane passed\n"); err != nil {
		t.Fatal(err)
	}
	for name, out := range map[string]string{
		"broad invocation skipped":   targeted + strings.ReplaceAll(broad, "\"pass\",\"Package\":\"overgo/internal/y\",\"Test\"", "\"skip\",\"Package\":\"overgo/internal/y\",\"Test\""),
		"selector matched nothing":   strings.ReplaceAll(targeted, "TestOne", "TestOther") + broad,
		"broad invocation absent":    targeted,
		"targeted invocation absent": broad,
	} {
		if err := VerifyGoTestEvidence(command, out+"lane passed\n"); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
