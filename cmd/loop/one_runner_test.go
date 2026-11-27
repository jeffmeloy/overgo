package main

import (
	"bytes"
	"path/filepath"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/processlock"
	"overgo/internal/testcheck"
)

// TestACloseDuringAMeasurementRunsWhenItLetsGo holds the cycle's lanes to one run at a time (lanes-record/one-runner):
// while a measurement's phases hold the lanes, a close's runner finds them held and runs nothing, the close having named
// its commit due; once the measurement lets go the runner starts for that commit, and a measurement no close came
// during starts none. The closing run of 2026-11-25 held no lock, and the runner a landing started ran the models lane
// beside it, the run waiting 130 times for the card.
func TestACloseDuringAMeasurementRunsWhenItLetsGo(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	started := 0
	next := func() error { started++; return nil }
	var output bytes.Buffer
	testcheck.NoError(t, holdingTheLanes(root, &output, func() error { return nil }, next))
	if started != 0 {
		t.Fatalf("a measurement no close came during started the runner %d times; want none", started)
	}
	due := filepath.Join(root, processlock.StateDirectory, lanesDueFile)
	testcheck.NoError(t, holdingTheLanes(root, &output, func() error {
		testcheck.NoError(t, jsonfile.Write(due, lanesDue{Commit: "newer", Rows: []string{"a close"}}, clioptions.PrivateFileMode))
		return runLanesDue(root, &output, func(string) error {
			t.Error("a close's runner ran the lanes while a measurement held them")
			return nil
		}, next)
	}, next))
	if started != 1 {
		t.Fatalf("a close during a measurement started the runner %d times; want once, when the measurement let the lanes go\n%s", started, output.String())
	}
}
