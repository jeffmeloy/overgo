// Worker fan-out for the host math primitives: chunked ranges over a
// persistent worker pool (spawning GOMAXPROCS goroutines per call dominated
// the wall on codec/backbone paths that issue thousands of small dispatches
// per second). Whether a call fans out at all, and across how many workers,
// is decided per call by the calibrated cost model in dispatch.go — measured
// host constants, no tuned thresholds. Chunks are independent, so pool
// scheduling is bit-identical to the serial loop.
package hostmath

import "sync"

type poolTask struct {
	fn         func(start, end int)
	start, end int
	done       *sync.WaitGroup
}

var (
	poolOnce  sync.Once
	poolTasks chan poolTask
)

func ensurePool() {
	poolOnce.Do(func() {
		workers := poolWorkerLimit() - 1
		if workers < 1 {
			workers = 1
		}
		poolTasks = make(chan poolTask, workers)
		for i := 0; i < workers; i++ {
			go func() {
				for task := range poolTasks {
					task.fn(task.start, task.end)
					task.done.Done()
				}
			}()
		}
	})
}

// parallelRangeCost: cost-modeled dispatch. unitMACs is the caller's per-unit
// work estimate (MACs of the rate's kernel class); dispatchWorkers turns it
// into serial-or-w-workers using the one-time host calibration.
func parallelRangeCost(n, unitMACs int, rate macRate, fn func(start, end int)) {
	workers := dispatchWorkers(n, unitMACs, rate)
	if workers < 2 {
		fn(0, n)
		return
	}
	fanOut(n, workers, fn)
}

// fanOut: unconditional chunked dispatch across workers (callers guarantee
// workers >= 2 and workers <= n). Caller works the first chunk; a saturated
// pool (nested or concurrent dispatch) runs the chunk inline.
func fanOut(n, workers int, fn func(start, end int)) {
	ensurePool()
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := chunk; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		task := poolTask{fn: fn, start: start, end: end, done: &wg}
		select {
		case poolTasks <- task:
		default:
			fn(start, end)
			wg.Done()
		}
	}
	fn(0, min(chunk, n))
	wg.Wait()
}
