package hostexpert

import (
	"sync"
	"sync/atomic"
)

// pool runs a batch of tasks across a fixed set of workers, as Strata's pool
// (src/kernels/cpu/pool.cpp) keeps a worker on each core: every worker takes
// the batch's next task until none is left, so a batch ends when its slowest
// task does.
type pool struct {
	workers int
	batches chan *batch
}

// batch is one run's tasks: how many, the next one to take, what a task
// does, the workers still in it and the first failure.
type batch struct {
	tasks   int64
	next    atomic.Int64
	run     func(task int) error
	working sync.WaitGroup
	failure atomic.Pointer[error]
}

// newPool starts workers workers, each until the pool is closed.
func newPool(workers int) *pool {
	p := &pool{workers: max(workers, 1), batches: make(chan *batch)}
	for range p.workers {
		go p.serve()
	}
	return p
}

func (p *pool) serve() {
	for current := range p.batches {
		for task := current.next.Add(1) - 1; task < current.tasks; task = current.next.Add(1) - 1 {
			if err := current.run(int(task)); err != nil {
				current.failure.CompareAndSwap(nil, &err)
			}
		}
		current.working.Done()
	}
}

// run runs every task from 0 to tasks across the workers, waits for them
// all and returns the first failure.
func (p *pool) run(tasks int, run func(task int) error) error {
	current := &batch{tasks: int64(tasks), run: run}
	current.working.Add(p.workers)
	for range p.workers {
		p.batches <- current
	}
	current.working.Wait()
	if failure := current.failure.Load(); failure != nil {
		return *failure
	}
	return nil
}

// close stops the workers once their batch is done.
func (p *pool) close() { close(p.batches) }
