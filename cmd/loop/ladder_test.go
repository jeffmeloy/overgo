package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// TestTheLocalWorkerGoesFirstWhereItsReplayPays holds the default configuration -init writes to the replays recorded
// beside it. For host steps under internal/ the local worker passed three of four in a sixth of the Claude worker's wall,
// so its attempts plus the Claude worker's on the one it failed cost less than the Claude worker alone, and it goes
// first; for steps under cmd/ it passed one of four, slower than the Claude worker, and for card steps none, so the
// Claude worker keeps those. A skipped replay counts neither way. With no replays recorded the Claude worker is the
// only worker.
func TestTheLocalWorkerGoesFirstWhereItsReplayPays(t *testing.T) {
	t.Parallel()
	const local = "Qwen3.8-27B-UD-Q6_K.gguf"
	var records []replayRecord
	// Each worker's records of a kind are its steps in order, so the two workers' nth records of a kind are one step.
	steps := map[string]int{}
	add := func(worker, kind string, passed bool, wall int64, times int) {
		for range times {
			step := kind + "/" + strconv.Itoa(steps[worker+" "+kind])
			steps[worker+" "+kind]++
			records = append(records, replayRecord{Worker: worker, Kind: kind, Step: step, Landed: "c0" + step, Passed: passed, Attempts: 1, WallMS: wall})
		}
	}
	add(local, "internal", true, 100_000, 3)
	add(local, "internal", false, 100_000, 1)
	add(defaultWorkerName, "internal", true, 600_000, 4)
	add(local, "cmd", true, 500_000, 1)
	add(local, "cmd", false, 500_000, 3)
	add(defaultWorkerName, "cmd", true, 300_000, 4)
	add(local, "internal+card", false, 200_000, 2)
	add(defaultWorkerName, "internal+card", true, 900_000, 2)
	records = append(records, replayRecord{Worker: local, Kind: "cmd", Passed: true, WallMS: 1, Skipped: "no task"})

	state := t.TempDir()
	lines := make([]string, len(records))
	for index, record := range records {
		lines[index] = string(testcheck.Must(json.Marshal(record))(t))
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(state, replayRecords), []byte(strings.Join(lines, "\n")+"\n"), 0o600))
	path := filepath.Join(state, "loop.json")
	testcheck.NoError(t, writeDefaultConfig(path, io.Discard))
	var written config
	testcheck.NoError(t, jsonfile.DecodeStrict(path, &written))
	t.Logf("-init wrote worker %v, then %v, first for %v", written.Worker, written.Workers, written.FirstFor)
	if !slices.Equal(written.FirstFor, []string{"internal"}) || !slices.Contains(written.Worker, local) || !slices.Contains(written.Worker, "-local-worker") ||
		len(written.Workers) != 1 || !slices.Equal(written.Workers[0], defaultWorker) {
		t.Fatalf("-init wrote worker %v, then %v, first for %v; want the local worker first for internal steps alone and the Claude worker after it", written.Worker, written.Workers, written.FirstFor)
	}
	for _, example := range []struct {
		files []string
		card  bool
		kind  string
	}{
		{[]string{"internal/evaluation", "cmd/evaluate"}, false, "internal"},
		{[]string{"cmd/loop"}, false, "cmd"},
		{nil, false, "any"},
		{[]string{"internal/latentvideo"}, true, "internal+card"},
	} {
		if got := stepKind(example.files, example.card); got != example.kind {
			t.Errorf("a step listing %v, on the card %t, is of kind %q, want %q", example.files, example.card, got, example.kind)
		}
	}

	bare := t.TempDir()
	testcheck.NoError(t, writeDefaultConfig(filepath.Join(bare, "loop.json"), io.Discard))
	var claudeOnly config
	testcheck.NoError(t, jsonfile.DecodeStrict(filepath.Join(bare, "loop.json"), &claudeOnly))
	if !slices.Equal(claudeOnly.Worker, defaultWorker) || len(claudeOnly.Workers) != 0 || len(claudeOnly.FirstFor) != 0 {
		t.Fatalf("with no replays -init wrote worker %v, then %v, first for %v; want the Claude worker alone", claudeOnly.Worker, claudeOnly.Workers, claudeOnly.FirstFor)
	}
}
