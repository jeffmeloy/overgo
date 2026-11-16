package processcontrol

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// jobBound is the commit a supervised tree may hold (newProcessTree): what its command's last run in this process
// that exited 0 needed, its whole tree's peak, and the commit free as it starts; no bound where this process has run the
// command to no pass, so a command's first run records its need. A runaway may not take more than its command once
// needed and the machine then had free: a replayed check run on the fault its step cured took 279 GB of commit at
// 10:05 on 2026-11-15, the page file grown to 256 GB, and a served model's read failed beside it (check-memory/do). A
// landing's verify and its undo run are one process, so a check undone repeats the command its verify has just
// recorded and is always bound. The need is kept in the process, not in files: every git command a landing runs
// starts here, and a file for each wrote 1,212 of them in one landing, racing to replace them.
type jobBound struct {
	key               string
	need, free, limit uint64
}

// jobKey names a command by what it runs, its program's name and its arguments, not where it runs: a check run in the
// candidate and again with its change undone is one command.
func jobKey(command *exec.Cmd) string {
	return strings.Join(append([]string{filepath.Base(command.Path)}, command.Args[1:]...), "\x00")
}

// jobNeeds holds each command's need as this process last recorded it (recordJobNeed).
var jobNeeds sync.Map

// readJobNeed is the need recorded for key in this process, where one is.
func readJobNeed(key string) (uint64, bool) {
	need, recorded := jobNeeds.Load(key)
	if !recorded {
		return 0, false
	}
	return need.(uint64), true
}

// recordJobNeed records peak as key's need.
func recordJobNeed(key string, peak uint64) { jobNeeds.Store(key, peak) }

// boundOf is command's bound as it starts: its recorded need and the commit free now, or none.
func boundOf(command *exec.Cmd) (jobBound, error) {
	bound := jobBound{key: jobKey(command)}
	need, recorded := readJobNeed(bound.key)
	if !recorded {
		return bound, nil
	}
	free, _, err := MachineCommit()
	if err != nil {
		return jobBound{}, fmt.Errorf("processcontrol: the commit free for a job's bound: %w", err)
	}
	bound.need, bound.free, bound.limit = need, free, need+free
	return bound, nil
}

// reached says the bound a tree reached, for the run's error.
func (bound jobBound) reached() error {
	gib := func(bytes uint64) float64 { return float64(bytes) / (1 << 30) }
	return fmt.Errorf("processcontrol: the tree reached its memory limit of %.1f GiB, the %.1f GiB its command's last passing run held and the %.1f GiB of commit free at its start",
		gib(bound.limit), gib(bound.need), gib(bound.free))
}
