package gate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGateRejectsSkippedCapabilityTest(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runGoTests(t.Context(), repo, []string{"./internal/testevidence/testdata/skipfixture"}, true); err == nil {
		t.Fatal("skipped capability test passed gate evidence")
	}
}

func TestGateAcceptsPassingCapabilityTest(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(repo, "go.mod")); err != nil {
		t.Fatal(err)
	}
	if _, err := runGoTests(t.Context(), repo, []string{"./internal/testevidence"}, true); err != nil {
		t.Fatal(err)
	}
}

func TestGateReportsUnchangedImporterSkipWithoutCreditingIt(t *testing.T) {
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	report, err := runGoTests(t.Context(), repo, []string{"./internal/testevidence/testdata/skipfixture"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Skipped) != 1 {
		t.Fatalf("report = %+v", report)
	}
}
