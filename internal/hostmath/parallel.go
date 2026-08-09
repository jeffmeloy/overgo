// Worker fan-out for the host math primitives: chunked ranges over a
// persistent worker pool (spawning GOMAXPROCS goroutines per call dominated
// the wall on codec/backbone paths that issue thousands of small dispatches
// per second). Serial when the range cannot feed at least two workers — a
// structural bound, not a tuned threshold. Chunks are independent, so pool
// scheduling is bit-identical to the serial loop.
package hostmath

import (
	"runtime"
	"sync"
)

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
		workers := runtime.GOMAXPROCS(0) - 1
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

func parallelRange(n int, fn func(start, end int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
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
			// Pool saturated (nested or concurrent dispatch): run inline.
			fn(start, end)
			wg.Done()
		}
	}
	fn(0, min(chunk, n)) // caller works the first chunk
	wg.Wait()
}
