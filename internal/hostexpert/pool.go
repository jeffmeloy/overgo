package hostexpert

import (
	"sync"
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
type Pool struct {
	workers int
	batches chan *batch
	// scratch is the layer's intermediate values, kept between layers.
	scratch layerScratch
	// spinning holds the workers polling current, the batch a spinning run
	// hands them; spinners counts the workers still polling.
	spinning atomic.Bool
	current  atomic.Pointer[batch]
	spinners sync.WaitGroup
}

// batch is one run's tasks: how many, the next one to take, what a task
// does, the workers still in it and the first failure. A spinning run counts
// its workers in left.
type batch struct {
	tasks   int64
	next    atomic.Int64
	run     func(task int) error
	working sync.WaitGroup
	left    atomic.Int64
	failure atomic.Pointer[error]
}

// spinBatch sends a worker to poll for batches until the pool stops spinning.
var spinBatch = &batch{}

// NewPool starts workers workers, each until the pool is closed.
func NewPool(workers int) *Pool {
	p := &Pool{workers: max(workers, 1), batches: make(chan *batch)}
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
		current.working.Done()
	}
}

// poll runs each batch a spinning run hands the workers, once, until the
// pool stops spinning.
func (p *Pool) poll() {
	defer p.spinners.Done()
	var last *batch
	for p.spinning.Load() {
		if current := p.current.Load(); current != nil && current != last {
			last = current
			current.take()
			current.left.Add(-1)
		}
	}
}

// take runs the batch's next task until none is left.
func (b *batch) take() {
	for task := b.next.Add(1) - 1; task < b.tasks; task = b.next.Add(1) - 1 {
		if err := b.run(int(task)); err != nil {
			b.failure.CompareAndSwap(nil, &err)
		}
	}
}

// Spin starts or stops the workers polling for batches. The goroutine that
// runs the pool's batches calls it, between them.
func (p *Pool) Spin(on bool) {
	if on == p.spinning.Load() {
		return
	}
	if !on {
		p.spinning.Store(false)
		p.spinners.Wait()
		p.current.Store(nil)
		return
	}
	p.spinning.Store(true)
	p.spinners.Add(p.workers)
	for range p.workers {
		p.batches <- spinBatch
	}
}

// run runs every task from 0 to tasks across the workers, waits for them
// all and returns the first failure.
func (p *Pool) run(tasks int, run func(task int) error) error {
	current := &batch{tasks: int64(tasks), run: run}
	if p.spinning.Load() {
		current.left.Store(int64(p.workers))
		p.current.Store(current)
		current.take()
		for current.left.Load() > 0 {
		}
	} else {
		current.working.Add(p.workers)
		for range p.workers {
			p.batches <- current
		}
		current.working.Wait()
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
