package planverify

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

// TestACheckThatNamesGoTestInItsTextRunsNone holds a check to what it runs (verify-mention/do): a check that greps
// text naming a go test, as a plan-only change's check reads the check it sets, runs none, so it passes on its own
// exit. It was refused as vacuous, a go test with no passing test, its pattern read as a command. A check that hands
// a shell a command string, where a go test would run unread, is refused before it runs.
func TestACheckThatNamesGoTestInItsTextRunsNone(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "plan.txt"), []byte("verify: go test ./internal/x -count=1 -run '^TestX$'\n"), 0o600))
	if _, err := Execute(t.Context(), directory, `grep -q "verify: go test ./internal/x" plan.txt`, nil); err != nil {
		t.Fatalf("a check that greps text naming go test was refused: %v", err)
	}
	if _, err := Execute(t.Context(), directory, `sh -c "grep -q verify plan.txt"`, nil); err == nil {
		t.Fatal("a check that hands a shell a command string ran")
	}
}
