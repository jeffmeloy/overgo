//go:build windows

package processcontrol

import (
	"os"
	"os/exec"
	"testing"

	"overgo/internal/testcheck"
)

// TestAHolderWhoseIDWasGivenOutAgainIsDead holds a holder's liveness to its id and its start
// (process-identity/start-time): this process, asked about by its id and its own start, runs; by its id and another
// start it is a process that took a dead holder's id, and the holder asked about has ended; by its id alone, as a record
// written before starts were kept, it runs; and a process that has exited runs by neither. On 2026-11-12 the 307th
// process started after one with id 13696 exited was given 13696, and ProcessAlive(13696) said it ran.
func TestAHolderWhoseIDWasGivenOutAgainIsDead(t *testing.T) {
	t.Parallel()
	pid := os.Getpid()
	start, read := ProcessStart(pid)
	if !read || start == 0 {
		t.Fatalf("this process's start reads %d (%t), want it read", start, read)
	}
	if !Running(pid, start) {
		t.Fatal("this process, asked about by its id and its start, is said to have ended")
	}
	if Running(pid, start-1) {
		t.Fatal("a holder whose id this process took is said to run")
	}
	if !Running(pid, 0) {
		t.Fatal("this process, asked about by its id alone, is said to have ended")
	}
	exited := exec.Command("cmd", "/c", "exit")
	testcheck.NoError(t, exited.Run())
	if Running(exited.Process.Pid, 0) || Running(exited.Process.Pid, start) {
		t.Fatal("a process that exited is said to run")
	}
}
