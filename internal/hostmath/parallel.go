// Worker fan-out for the host math primitives: chunked ranges over
// GOMAXPROCS goroutines. Serial when the range cannot feed at least two
// workers — a structural bound, not a tuned threshold.
package hostmath

import (
	"runtime"
	"sync"
)

func parallelRange(n int, fn func(start, end int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n/2 {
		workers = n / 2
	}
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}
