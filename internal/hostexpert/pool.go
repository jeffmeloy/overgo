package hostexpert

import (
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
	batches chan *batch
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
		go p.serve()
	}
	return p
}

func (p *Pool) serve() {
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
