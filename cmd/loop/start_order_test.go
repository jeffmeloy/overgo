package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestARefusedSecondDriverLeavesTheRunningRecordUntouched holds a refused start to the state it found (SQA-338): the
// driver logged a live run as unclean and removed its running record before the campaign lock refused it. With a
// campaign held and a running record kept, a second start is refused, the record keeps its bytes and no unclean end
// is logged.
func TestARefusedSecondDriverLeavesTheRunningRecordUntouched(t *testing.T) {
	// Serial: it changes the process's directory and environment.
	t.Chdir(t.TempDir())
	t.Setenv(plan.AutomationWorkerEnvironment, "")
	testcheck.NoError(t, run([]string{"-init"}))
	first := testcheck.Must(processcontrol.BeginCampaign(loopWorktree, defaultWorkerName))(t)
	defer first.Close()
	testcheck.NoError(t, keepRunning(loopWorktree, "row/do", time.Now()))
	record := filepath.Join(processlock.StateDirectory, runningRecordFile)
	kept := testcheck.Must(os.ReadFile(record))(t)
	if err := run([]string{"-repo", filepath.Join(t.TempDir(), "absent-store")}); err == nil {
		t.Fatal("a second driver started beside a running one")
	}
	if now, err := os.ReadFile(record); err != nil || string(now) != string(kept) {
		t.Fatalf("the refused start left the running record %q (%v), want %q", now, err, kept)
	}
	if _, err := os.Stat(filepath.Join(processlock.StateDirectory, uncleanLogFile)); err == nil {
		t.Fatal("the refused start logged the running worker's run as an unclean end")
	}
}
