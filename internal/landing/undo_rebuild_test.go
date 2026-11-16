package landing

import (
	"testing"
	"testing/synctest"
	"time"
)

// TestAnUndoRunsRebuildIsNoWait holds the undo run's clock to its tests, not their build (undo-overrun/do): on
// 2026-11-16 three undo runs that rebuilt the restored production and failed within seconds were each said to wait on
// what only the change sends, their bound twice a run whose build the stages had made. On a simulated clock, the
// clock says nothing however long the run builds before its first test binary starts; once the tests have begun it
// says them at the bound and not before, once; and tests that end before the bound are never said.
func TestAnUndoRunsRebuildIsNoWait(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const bound = time.Minute
		said := 0
		began, stop := overrunClock(bound, func() { said++ })
		time.Sleep(10 * bound)
		synctest.Wait()
		if said != 0 {
			t.Fatalf("a run building for ten times its bound before its tests began was said %d times; want none", said)
		}
		began()
		time.Sleep(bound / 2)
		synctest.Wait()
		if said != 0 {
			t.Fatal("tests were said half their bound after they began")
		}
		began()
		time.Sleep(bound)
		synctest.Wait()
		if said != 1 {
			t.Fatalf("tests that outlasted their bound were said %d times; want once, a second package's start arming nothing", said)
		}
		stop()

		quick, ended := overrunClock(bound, func() { said++ })
		quick()
		time.Sleep(bound / 2)
		ended()
		time.Sleep(10 * bound)
		synctest.Wait()
		if said != 1 {
			t.Fatal("tests that ended before their bound were said")
		}
	})
}
