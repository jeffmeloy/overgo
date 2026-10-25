package hostexpert

import (
	"errors"
	"sync/atomic"
	"testing"
)

// TestPoolRunsWithoutItsWorkers holds a pool's caller to never waiting for a worker to be scheduled: a pool whose
// workers never run still runs every task of a run, spinning or not, with its first failure, and starts and stops
// spinning, all on the caller alone. A pool whose caller waits for its workers never returns here.
func TestPoolRunsWithoutItsWorkers(t *testing.T) {
	const workers, tasks = 4, 10
	p := &Pool{workers: workers, batches: make(chan *batch, poolWakeups*workers)}
	for _, spin := range []bool{false, true, true, false} {
		p.Spin(spin)
		var ran atomic.Int64
		failure := errors.New("task 3")
		err := p.run(tasks, func(task int) error {
			ran.Add(1)
			if task == 3 {
				return failure
			}
			return nil
		})
		if !errors.Is(err, failure) || ran.Load() != tasks {
			t.Fatalf("a run without its workers ran %d of %d tasks and answered %v", ran.Load(), tasks, err)
		}
	}
	p.Spin(false)
}
