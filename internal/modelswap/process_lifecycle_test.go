package modelswap

import (
	"runtime"
	"testing"
	"time"

	"overgo/internal/processcontrol"
)

// TestExternalProcessLifecycle proves the launcher child type rides
// the shared supervisor: a long-running child under containment is
// terminated and reaped by Stop, tree included.
func TestExternalProcessLifecycle(t *testing.T) {
	path, args := "sh", []string{"-c", "sleep 60"}
	if runtime.GOOS == "windows" {
		path, args = "cmd", []string{"/c", "ping -n 60 127.0.0.1 >nul"}
	}
	supervised, err := processcontrol.Start(t.Context(), processcontrol.Command{Path: path, Args: args})
	if err != nil {
		t.Fatal(err)
	}
	exited := make(chan error, 1)
	go func() {
		_, waitErr := supervised.Wait(t.Context())
		exited <- waitErr
		close(exited)
	}()
	process := &serverProcess{supervised: supervised, exited: exited, url: "http://127.0.0.1:0"}
	start := time.Now()
	if err := process.Stop(); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 10*time.Second {
		t.Fatal("stop did not terminate the child promptly")
	}
	if !supervised.Exited() {
		t.Fatal("stop returned before the tree reached a terminal state")
	}
}
