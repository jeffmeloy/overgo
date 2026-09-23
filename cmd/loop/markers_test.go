package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// TestLoopMarkersLiveInTheRuntimeDirectory holds the loop's pause marker and
// default config to the checkout's process state: a marker or config left
// under docs is carried across once, the carried marker still pauses the
// loop, and docs keeps neither.
func TestLoopMarkersLiveInTheRuntimeDirectory(t *testing.T) {
	t.Chdir(t.TempDir())
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	for _, legacy := range []string{legacyPauseMarker, processcontrol.LegacyLoopConfigFile} {
		if err := os.WriteFile(filepath.FromSlash(legacy), []byte("{}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if !(&execWorld{}).Paused() {
		t.Fatal("a pause marker left under docs no longer pauses the loop")
	}
	config, err := processlock.StateFile(".", processcontrol.LoopConfigFile, processcontrol.LegacyLoopConfigFile)
	if err != nil || config != filepath.Join(processlock.StateDirectory, processcontrol.LoopConfigFile) {
		t.Fatalf("loop config at %q: %v", config, err)
	}
	for _, carried := range []string{pauseMarker, processcontrol.LoopConfigFile} {
		if _, err := os.Stat(filepath.Join(processlock.StateDirectory, carried)); err != nil {
			t.Fatalf("%s was not carried into the state directory: %v", carried, err)
		}
	}
	for _, legacy := range []string{legacyPauseMarker, processcontrol.LegacyLoopConfigFile} {
		if _, err := os.Stat(filepath.FromSlash(legacy)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s stayed under docs: %v", legacy, err)
		}
	}
}
