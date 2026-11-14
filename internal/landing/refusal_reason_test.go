package landing

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// TestARefusalSaysItsReasonBeforeItsCommand holds a refused command's record to its reason (the owner's review,
// 2026-11-14): a refusal is recorded by its first line cut to causeBytes, and a command's failure opened with the
// command, so four device lane refusals on record held only "go run ./cmd/device-lane -paths-file ...: exit 1". A lane
// whose step fails with output that is no test's, ending on the line that names the failed step, and run with a paths
// file long enough to fill the line alone, is recorded by that last line.
func TestARefusalSaysItsReasonBeforeItsCommand(t *testing.T) {
	const child, failed = "OVERGO_LANDING_REASON_CHILD", "device-lane: go test ./internal/cuda/executor failed"
	if os.Getenv(child) != "" {
		fmt.Println("[device] step 2 of 3: go test ./internal/cuda/executor")
		fmt.Fprintln(os.Stderr, failed)
		os.Exit(1)
	}
	t.Setenv(child, "1")
	paths := strings.Repeat("C:/Users/someone/AppData/Local/Temp/", 8) + "overgo-landing-paths.txt"
	_, err := (&landing{}).runIn(t.Context(), t.TempDir(), os.Environ(), os.Args[0], "-test.run=^TestARefusalSaysItsReasonBeforeItsCommand$", "lane", "-paths-file", paths)
	if err == nil {
		t.Fatal("the failing lane's command was taken")
	}
	if refusal := Classify("row/do", "gate: land: lanes: lane device: "+testFailure("", err).Error()); !strings.Contains(refusal.Cause, failed) {
		t.Errorf("the lane's refusal is recorded as %q; want its cause to hold the line the lane failed on, %q", refusal.Cause, failed)
	}
}
