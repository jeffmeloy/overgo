package processcontrol

import (
	"fmt"
	"os/exec"
)

// jobBound is the commit a supervised tree may hold (newProcessTree): the machine's commit limit as the tree starts,
// or twice the memory installed in it where that is more, so a tree may grow the page file as far as a page file the
// size of the machine's memory. A runaway may not take more: a replayed check run on the fault its step cured took 279
// GB of commit at 10:05 on 2026-11-15, the page file grown to 256 GB, and a served model's read failed beside it
// (check-memory/do). A tree that grows the page file by design may: Flash-Next's open reserves its commit with an
// ordinary commit, which Windows grows the page file for (flash-next-load), and its tests' trees peak at 115.6 GiB on a
// machine of 96 GiB whose commit limit was 99.6 GiB at rest, every one of them ended there (flash-next-bound). The
// bound reads nothing another process holds, so a check's verdict is the same whatever runs beside it
// (verdict-steady): bounded by the commit free as it started, the cycle's lanes at 1ab50872c ended at 20.7 GiB on
// 2026-11-19 while a model server another lane had started held the rest.
type jobBound struct {
	limit uint64
}

// boundOf is command's bound as it starts: the machine's commit limit, or twice its memory where that is more.
func boundOf(*exec.Cmd) (jobBound, error) {
	_, limit, err := MachineCommit()
	if err != nil {
		return jobBound{}, fmt.Errorf("processcontrol: the commit limit for a job's bound: %w", err)
	}
	memory, err := machineMemory()
	if err != nil {
		return jobBound{}, fmt.Errorf("processcontrol: the memory for a job's bound: %w", err)
	}
	return jobBound{limit: max(limit, 2*memory)}, nil
}

// reached says the bound a tree reached, for the run's error.
func (bound jobBound) reached() error {
	return fmt.Errorf("processcontrol: the tree reached its memory limit of %.1f GiB, the machine's commit limit as it started or twice its memory",
		float64(bound.limit)/(1<<30))
}
