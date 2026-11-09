package main

import (
	"bytes"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALandingRecordsItsFullCost reads the cost each landing says back from its trailers and prints, for each worker,
// the steps its landings landed against what they cost, as the owner's review (2026-11-09) asked: benefit against the
// full cost, per worker. Of four landings, two by the local worker land three steps in 4 attempts, 900 s of their wall
// and 120 s of the gate, one names no recorded attempt, and one is from before landings said their cost, which is
// counted apart. A worker is named by its configured name, a later one by the local model it serves, else by its
// command.
func TestALandingRecordsItsFullCost(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	git := func(arguments ...string) {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", arguments...)
		command.Dir = root
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
	}
	git("init", "-q")
	git("config", "user.email", "fixture@example.invalid")
	git("config", "user.name", "fixture")
	for _, message := range []string{
		"a/old: before costs\n\nLanded-Row: a/old",
		"a/one: one step\n\nCost: said\n\nLanded-Row: a/one\nWorker-Attempts: 3\nWorker-Wall-Seconds: 600\nGate-Wall-Seconds: 60\nWorker: local qwen",
		"a/two,a/three: two steps\n\nLanded-Row: a/two\nLanded-Row: a/three\nWorker-Attempts: 1\nWorker-Wall-Seconds: 300\nGate-Wall-Seconds: 60\nWorker: local qwen",
		"a/four: by hand\n\nLanded-Row: a/four\nWorker-Attempts: 0\nWorker-Wall-Seconds: 0\nGate-Wall-Seconds: 45",
	} {
		git("commit", "-q", "--allow-empty", "-m", message)
	}
	costs, unsaid, err := landingCosts(t.Context(), root, "2026-11-01")
	testcheck.NoError(t, err)
	var out bytes.Buffer
	printCosts(&out, costs, unsaid)
	report := out.String()
	for _, want := range []string{
		"local qwen: landed 3 steps in 2 landings, 4 worker attempts in 15m0s of their wall, the gate 2m0s\n",
		"no recorded attempt: landed 1 steps in 1 landings, 0 worker attempts in 0s of their wall, the gate 45s\n",
		"1 landings from before landings said their cost are not counted\n",
	} {
		if !strings.Contains(report, want) {
			t.Errorf("the report reads\n%s\nwant a line %q", report, want)
		}
	}
	for command, want := range map[string]string{
		"claude -p": "claude-master-lead",
		"go run ./cmd/loop -local-worker x -local-model qwen": "local qwen",
		"C:/tools/worker.exe -p":                              "worker",
	} {
		level := 1
		if want == "claude-master-lead" {
			level = 0
		}
		if got := workerName("claude-master-lead", level, strings.Fields(command)); got != want {
			t.Errorf("the worker %q at level %d is named %q, want %q", command, level, got, want)
		}
	}
}
