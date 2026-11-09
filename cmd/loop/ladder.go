package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/loop"
	"overgo/internal/plan"
)

// replayRecords is the file in the process state each replay's records are added to, which -init reads the ladder from.
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
// (config.FirstFor), the worker after it for every other kind where the ladder has one, so a kind no replay showed the
// first worker paying for goes to the stronger worker first.
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

// localFirst are the kinds of step the local worker should take first, read from replay records: those where its
// expected wall, its own attempts plus the strong worker's on the share of steps it fails, is below the strong worker's
// alone. A kind the local worker passed none of, or the strong worker was not replayed on, stays with the strong worker.
func localFirst(records []replayRecord, local, strong string) []string {
	type tally struct {
		passed, tasks int
		wall          int64
	}
	byKind := map[string]map[string]*tally{}
	for _, record := range records {
		if record.Skipped != "" || record.Worker != local && record.Worker != strong {
			continue
		}
		if byKind[record.Kind] == nil {
			byKind[record.Kind] = map[string]*tally{}
		}
		counted := byKind[record.Kind][record.Worker]
		if counted == nil {
			counted = &tally{}
			byKind[record.Kind][record.Worker] = counted
		}
		counted.tasks++
		counted.wall += record.WallMS
		if record.Passed {
			counted.passed++
		}
	}
	var kinds []string
	for kind, workers := range byKind {
		weak, strongest := workers[local], workers[strong]
		if weak == nil || strongest == nil || weak.passed == 0 {
			continue
		}
		strongWall := float64(strongest.wall) / float64(strongest.tasks)
		failing := 1 - float64(weak.passed)/float64(weak.tasks)
		if float64(weak.wall)/float64(weak.tasks)+failing*strongWall < strongWall {
			kinds = append(kinds, kind)
		}
	}
	slices.Sort(kinds)
	return kinds
}

// readReplayRecords reads the replay records at path, none where there is no file.
func readReplayRecords(path string) ([]replayRecord, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var records []replayRecord
	lines := bufio.NewScanner(file)
	for lines.Scan() {
		var record replayRecord
		if err := json.Unmarshal(lines.Bytes(), &record); err != nil {
			return nil, err
		}
		records = append(records, record)
	}
	return records, lines.Err()
}

// ladderFromReplays sets the default configuration's ladder from the replay records beside it: where they show the
// local worker paying for some kind of step, the local worker becomes the first worker, taking those kinds first, and
// the Claude worker the one after it.
func ladderFromReplays(defaults *config, records []replayRecord) {
	local := ""
	for _, record := range records {
		if record.Worker != defaultWorkerName && record.Skipped == "" {
			local = record.Worker
		}
	}
	if kinds := localFirst(records, local, defaultWorkerName); local != "" && len(kinds) != 0 {
		defaults.Worker = []string{"go", "run", "./cmd/loop", "-local-worker", localServerAddress, "-local-model", local}
		defaults.Workers = [][]string{defaultWorker}
		defaults.FirstFor = kinds
	}
}
