package webuilane

import (
	"context"
	"runtime"
	"testing"

	"overgo/internal/processcontrol"
)

// TestExternalProcessLifecycle proves the lane detects a supervised
// browser that exits before publishing its DevTools port: the exit is
// observed through the supervisor, not a poll of a raw process.
func TestExternalProcessLifecycle(t *testing.T) {
	path, args := "sh", []string{"-c", "exit 0"}
	if runtime.GOOS == "windows" {
		path, args = "cmd", []string{"/c", "exit 0"}
	}
	supervised, err := processcontrol.Start(context.Background(), processcontrol.Command{Path: path, Args: args})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := waitDevToolsPort(context.Background(), t.TempDir(), supervised); err == nil {
		t.Fatal("exited browser reported a DevTools port")
	}
}
