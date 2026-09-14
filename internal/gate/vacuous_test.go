package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

func TestGateRejectsSkippedCapabilityTest(t *testing.T) {
	t.Parallel()
	if _, err := liveRepositoryFixture(t).context().runGoTests(t.Context(), []string{"./internal/testevidence/testdata/skipfixture"}, true, nil); err == nil {
		t.Fatal("skipped capability test passed gate evidence")
	}
}

func TestGateAcceptsPassingCapabilityTest(t *testing.T) {
	t.Parallel()
	if _, err := liveRepositoryFixture(t).context().runGoTests(t.Context(), []string{"./internal/testevidence"}, true, nil); err != nil {
		t.Fatal(err)
	}
}

func TestGateReportsUnchangedImporterSkipWithoutCreditingIt(t *testing.T) {
	t.Parallel()
	report, err := liveRepositoryFixture(t).context().runGoTests(t.Context(), []string{"./internal/testevidence/testdata/skipfixture"}, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Skipped) != 1 {
		t.Fatalf("report = %+v", report)
	}
}

func TestGateNamesFailureBeforeDiagnosticTail(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	runGitFixture(t, repo, "init")
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module failurefixture\n\ngo 1.26\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// A later passing test pushes the early failure out of command's log tail.
	source := "package failurefixture\nimport \"testing\"\n" +
		"func TestEarlyFailure(t *testing.T) { t.Fatal(\"expected failure\") }\n" +
		"func TestLaterOutput(t *testing.T) { t.Log(\"" + strings.Repeat("x", clioptions.DiagnosticTailBytes*2) + "\") }\n"
	if err := os.WriteFile(filepath.Join(repo, "failure_test.go"), []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := (&gateContext{repo: repo}).runGoTests(t.Context(), []string{"."}, false, nil); err == nil || !strings.Contains(err.Error(), "failurefixture: TestEarlyFailure") {
		t.Fatalf("early failure identity was lost: %v", err)
	}
}
