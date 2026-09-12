package webuilane

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"overgo/internal/processcontrol"
)

func TestBrowserCloseReportsCleanupFailure(t *testing.T) {
	foreign := t.TempDir()
	marker := filepath.Join(foreign, "keep")
	if err := os.WriteFile(marker, []byte("unowned"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := (&Browser{profile: foreign}).Close(); err == nil {
		t.Fatal("unowned cleanup reported success")
	}
	if data, err := os.ReadFile(marker); err != nil || string(data) != "unowned" {
		t.Fatalf("unowned profile changed: %q, %v", data, err)
	}
	owned, err := os.MkdirTemp("", "overgo-webui-lane-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(owned)
	browser := &Browser{profile: owned}
	if err := browser.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(owned); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("owned profile remains: %v", err)
	}
	if err := browser.Close(); err != nil {
		t.Fatalf("repeated close: %v", err)
	}
}

// TestExternalProcessLifecycle proves the lane detects a supervised
// browser that exits before publishing its DevTools port: the exit is
// observed through the supervisor, not a poll of a raw process.
func TestExternalProcessLifecycle(t *testing.T) {
	path, args := "sh", []string{"-c", "exit 0"}
	if runtime.GOOS == "windows" {
		path, args = "cmd", []string{"/c", "exit 0"}
	}
	supervised, err := processcontrol.Start(t.Context(), processcontrol.Command{Path: path, Args: args})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := waitDevToolsPort(t.Context(), t.TempDir(), supervised); err == nil {
		t.Fatal("exited browser reported a DevTools port")
	}
}
