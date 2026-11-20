package processcontrol

import (
	"fmt"
	"os/exec"
)

// jobBound is the commit a supervised tree may hold (newProcessTree): the machine's commit limit as the tree starts,
// the most the machine can commit without growing its page file. A runaway may not take more: a replayed check run on
// the fault its step cured took 279 GB of commit at 10:05 on 2026-11-15, the page file grown to 256 GB, and a served
// model's read failed beside it (check-memory/do). The bound reads nothing another process holds, so a check's verdict
// is the same whatever runs beside it (verdict-steady): bounded by the commit free as it started, the cycle's lanes at
// 1ab50872c ended at 20.7 GiB on 2026-11-19 while a model server another lane had started held the rest.
type jobBound struct {
	limit uint64
}

// boundOf is command's bound as it starts: the machine's commit limit.
func boundOf(*exec.Cmd) (jobBound, error) {
	_, limit, err := MachineCommit()
	if err != nil {
		return jobBound{}, fmt.Errorf("processcontrol: the commit limit for a job's bound: %w", err)
	}
	return jobBound{limit: limit}, nil
}

// reached says the bound a tree reached, for the run's error.
func (bound jobBound) reached() error {
	return fmt.Errorf("processcontrol: the tree reached its memory limit of %.1f GiB, the machine's commit limit as it started",
		float64(bound.limit)/(1<<30))
}
