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

// localFirst are the kinds of step the local worker should take first, read from replay records: those where the
// ladder's wall on the steps both workers were replayed on is below the strong worker's alone on the same steps. A kind
// the local worker passed none of, or holding a step the strong worker was not replayed on, stays with the strong
// worker.
func localFirst(records []replayRecord, local, strong string) []string {
	var kinds []string
	for kind, spent := range cascades(records, local, strong) {
		if !spent.unmatched && spent.passed > 0 && spent.ladder < spent.alone {
			kinds = append(kinds, kind)
		}
	}
	slices.Sort(kinds)
	return kinds
}

// cascade is what one kind of step cost on the steps both workers were replayed on: the ladder's wall, the local
// worker's and the strong worker's on the steps the local worker failed, against the strong worker's alone.
type cascade struct {
	ladder, alone int64
	steps, passed int
	// unmatched is a step the local worker was replayed on and the strong worker was not, which leaves the kind with
	// no comparison.
	unmatched bool
}

// cascades reads the replay records by kind of step, each step matched by its reference and the commit it landed on
// with each worker's last record of it, so a step's cost is set against the same step's. Averages over the steps
// apart put the local worker first where it failed the hard steps, the strong worker's long ones (SQA-348). A step
// either worker skipped counts neither way.
func cascades(records []replayRecord, local, strong string) map[string]*cascade {
	type task struct{ kind, step, landed string }
	last := map[string]map[task]replayRecord{local: {}, strong: {}}
	for _, record := range records {
		if record.Skipped != "" || last[record.Worker] == nil {
			continue
		}
		last[record.Worker][task{record.Kind, record.Step, record.Landed}] = record
	}
	byKind := map[string]*cascade{}
	for at, weak := range last[local] {
		if byKind[at.kind] == nil {
			byKind[at.kind] = &cascade{}
		}
		spent := byKind[at.kind]
		strongest, matched := last[strong][at]
		if !matched {
			spent.unmatched = true
			continue
		}
		spent.steps++
		spent.ladder += weak.WallMS
		spent.alone += strongest.WallMS
		if weak.Passed {
			spent.passed++
		} else {
			spent.ladder += strongest.WallMS
		}
	}
	return byKind
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
