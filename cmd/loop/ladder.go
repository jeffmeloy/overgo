package main

import (
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/loop"
	"overgo/internal/plan"
)

// replayRecords is the file in the process state each replay's records are added to, the local worker's history.
const replayRecords = "replay-records.jsonl"

// localServerAddress is where the local worker finds overgo's server unless its configuration says otherwise: the
// server's own default listen address.
const localServerAddress = "http://127.0.0.1:8080"

// stepKind is the kind of step a worker's replayed pass rate is read by: the top directory of the first file it lists,
// any where it lists none, and whether it needs the card.
func stepKind(files []string, card bool) string {
	kind := "any"
	if len(files) != 0 {
		kind, _, _ = strings.Cut(filepath.ToSlash(files[0]), "/")
	}
	if card {
		return kind + "+card"
	}
	return kind
}

// StartLevel is the rung a step starts at (loop.StartingWorld): the first worker for the kinds it takes first
// (config.FirstFor), the worker after it for every other kind where the configuration names one (config.Workers).
func (w *execWorld) StartLevel(step loop.Step) int {
	if len(w.config.Workers) == 0 {
		return 0
	}
	document, err := plan.Load("")
	held, found := plan.ExactStep(document, step.Item, step.ID)
	if err == nil && found && slices.Contains(w.config.FirstFor, stepKind(held.Files, held.NeedsCard)) {
		return 0
	}
	return 1
}
