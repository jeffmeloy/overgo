package main

import (
	"slices"
	"testing"
)

// TestTheLadderSetsEachStepAgainstTheSameStep: the local worker goes first for a kind of step only where, step by step
// on the steps both workers were replayed on, its wall and the Claude worker's on the steps it failed come to less than
// the Claude worker's alone (SQA-348). It passes nine of ten scripts steps as fast as the Claude worker and fails the
// tenth, the one that takes the Claude worker ten times as long: averages over the kind put it first, though the
// ladder spends 20 s there where the Claude worker alone spends 19. It passes two docs steps fast, and the Claude
// worker was replayed on only the first, so the kind has no comparison. Of the tools steps it passes the one the Claude
// worker took a minute on and fails a short one, so the ladder pays there.
func TestTheLadderSetsEachStepAgainstTheSameStep(t *testing.T) {
	t.Parallel()
	const local = "Qwen3.8-27B-UD-Q6_K.gguf"
	record := func(worker, kind, step string, passed bool, wall int64) replayRecord {
		return replayRecord{Worker: worker, Kind: kind, Step: kind + "/" + step, Landed: "c0" + step, Passed: passed, Attempts: 1, WallMS: wall}
	}
	var records []replayRecord
	for _, step := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"} {
		records = append(records, record(local, "scripts", step, true, 1_000), record(defaultWorkerName, "scripts", step, true, 1_000))
	}
	records = append(records,
		record(local, "scripts", "j", false, 1_000), record(defaultWorkerName, "scripts", "j", true, 10_000),
		record(local, "docs", "a", true, 1_000), record(local, "docs", "b", true, 1_000), record(defaultWorkerName, "docs", "a", true, 100_000),
		record(local, "tools", "a", true, 5_000), record(defaultWorkerName, "tools", "a", true, 60_000),
		record(local, "tools", "b", false, 5_000), record(defaultWorkerName, "tools", "b", true, 10_000),
	)
	if got := localFirst(records, local, defaultWorkerName); !slices.Equal(got, []string{"tools"}) {
		t.Fatalf("the local worker goes first for %v, want tools alone: scripts costs the ladder 20 s against 19, docs has no comparison", got)
	}
}
