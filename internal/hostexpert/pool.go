package hostexpert

import (
	"runtime"
	"sync/atomic"
)

// Pool runs a batch of tasks across a fixed set of workers, as Strata's pool
// (src/kernels/cpu/pool.cpp) keeps a worker on each core: every worker takes
// the batch's next task until none is left, so a batch ends when its slowest
// task does.
//
// While it spins (Spin), its workers poll for the next batch instead of
// waiting on a channel, and the caller takes tasks beside them: a decode
// step's experts on host are due within microseconds of the card's ring,
// where waking a worker takes about as long as one expert's rows.
//
// The caller never waits for a worker to be scheduled: it wakes the workers
// without blocking, takes tasks itself, and a batch ends when its tasks have
// run. Handing a batch to each worker in turn over an unbuffered channel held
// the caller until every worker ran: a Flash-Next decode window's doorbell
// waited about 1.6 ms a window to start its workers spinning, and a prompt's
// layer of CPU experts behind every worker's turn.
type Pool struct {
	workers int
	// tasksPerThread splits a phase into that many tasks for each worker and the caller (NewCorePool); none splits
	// it into a task a worker.
	tasksPerThread int
	batches        chan *batch
	// scratch is the layer's intermediate values, kept between layers.
	scratch layerScratch
	// spinning holds the workers polling current, the batch a spinning run
	// hands them.
	spinning atomic.Bool
	current  atomic.Pointer[batch]
}

// batch is one run's tasks: how many, the next one to take, how many have
// run, what a task does and the first failure.
type batch struct {
	tasks   int64
	next    atomic.Int64
	done    atomic.Int64
	run     func(task int) error
	failure atomic.Pointer[error]
}

// spinBatch sends a worker to poll for batches until the pool stops spinning.
var spinBatch = &batch{}

// poolWakeups is the batches a pool's channel holds a worker, so a run or a
// spin wakes every worker without waiting for one to be scheduled.
const poolWakeups = 4

// NewPool starts workers workers, each until the pool is closed.
func NewPool(workers int) *Pool {
	workers = max(workers, 1)
	p := &Pool{workers: workers, batches: make(chan *batch, poolWakeups*workers)}
	for range p.workers {
		go p.serve(noCore)
	}
	return p
}

// coreTasksPerThread is Strata's tasks a phase for each of its pool's threads (phase_tasks): with a few tasks each, a
// core the system slows takes fewer of them, where a task a thread ends the phase with the slowest.
const coreTasksPerThread = 3

// coreLogicalShare is the logical processors a worker takes where the system names no cores: the two threads of an
// SMT core.
const coreLogicalShare = 2

// noCore is a worker the system places.
const noCore = -1

// NewCorePool starts Strata's expert pool (src/kernels/cpu/pool.cpp, ExpertPool with its defaults): a worker on each
// physical core but the first, which it leaves to the caller's thread, each worker's thread held to its core's first
// logical processor, and each phase split into coreTasksPerThread tasks a thread. A Flash-Next decode window's layer of
// two host experts of three tokens took 134 us on 16 unpinned workers with a task each, 100 us so, and a chat decoded
// about 2% faster. Where the system names no cores, a worker on each pair of logical processors, placed by the system.
func NewCorePool() *Pool {
	cores := physicalCores()
	workers := max(1, len(cores)-1)
	if len(cores) == 0 {
		workers = max(1, runtime.NumCPU()/coreLogicalShare)
	}
	p := &Pool{workers: workers, tasksPerThread: coreTasksPerThread, batches: make(chan *batch, poolWakeups*workers)}
	for index := range workers {
		core := noCore
		if index+1 < len(cores) {
			core = cores[index+1]
		}
		go p.serve(core)
	}
	return p
}

// serve runs the batches a worker is handed, its thread held to core unless it is noCore.
func (p *Pool) serve(core int) {
	if core != noCore {
		pinThread(core)
	}
	for current := range p.batches {
		if current == spinBatch {
			p.poll()
			continue
		}
		current.take()
	}
}

// poll runs each batch a spinning run hands the workers until the pool stops
// spinning.
func (p *Pool) poll() {
	var last *batch
	for p.spinning.Load() {
		if current := p.current.Load(); current != nil && current != last {
			last = current
			current.take()
		}
	}
}

// take runs the batch's next task until none is left; a batch taken after
// its tasks ran runs none.
func (b *batch) take() {
	for task := b.next.Add(1) - 1; task < b.tasks; task = b.next.Add(1) - 1 {
		if err := b.run(int(task)); err != nil {
			b.failure.CompareAndSwap(nil, &err)
		}
		b.done.Add(1)
	}
}

// wake hands batch to every worker without waiting for one; a worker the
// channel holds no room for is still busy, and finds no task left.
func (p *Pool) wake(batch *batch) {
	for range p.workers {
		select {
		case p.batches <- batch:
		default:
			return
		}
	}
}

// Spin starts or stops the workers polling for batches. The goroutine that
// runs the pool's batches calls it, between them.
func (p *Pool) Spin(on bool) {
	if on == p.spinning.Load() {
		return
	}
	p.spinning.Store(on)
	if !on {
		p.current.Store(nil)
		return
	}
	p.wake(spinBatch)
}

// run runs every task from 0 to tasks across the workers and the caller,
// waits until every task has run and returns the first failure.
func (p *Pool) run(tasks int, run func(task int) error) error {
	current := &batch{tasks: int64(tasks), run: run}
	if p.spinning.Load() {
		p.current.Store(current)
	} else {
		p.wake(current)
	}
	current.take()
	for current.done.Load() < current.tasks {
	}
	if failure := current.failure.Load(); failure != nil {
		return *failure
	}
	return nil
}

// Close stops the workers once their batch is done.
func (p *Pool) Close() {
	p.Spin(false)
	close(p.batches)
}
