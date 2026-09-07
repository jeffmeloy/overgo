package server

import (
	"sync"
	"testing"
)

// TestInflightRegistryConcurrentFinish exercises completion while admission
// scans the retained turns. Run with -race to check the two lock domains.
func TestInflightRegistryConcurrentFinish(t *testing.T) {
	for range 100 {
		var registry inflightRegistry
		before := registry.begin("before", 1)
		var during *inflightTurn
		var workers sync.WaitGroup
		workers.Go(func() { before.finish(nil, "") })
		workers.Go(func() { during = registry.begin("during", 1) })
		workers.Wait()
		during.finish(nil, "")
		last := registry.begin("last", 1)
		if got, found := registry.lookup("last"); !found || got != last {
			t.Fatal("live turn lost during concurrent completion")
		}
		if len(registry.turns) != 1 {
			t.Fatalf("completed turns retained: %d", len(registry.turns))
		}
		last.finish(nil, "")
	}
}
