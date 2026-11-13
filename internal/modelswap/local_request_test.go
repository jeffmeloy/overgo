package modelswap

import (
	"testing"
	"testing/synctest"
	"time"

	"overgo/internal/testcheck"
)

// TestALocalRequestEndsWhenItsRequesterEndsIt holds a swap to the old child's requests: one still answering is waited
// for however long it runs, an hour of the bubble's clock among them, and the old child stops and the new model
// launches once its requester releases it (no-clocks/local-serving). A 15-second grace had stopped the old child under
// a stream still answering.
func TestALocalRequestEndsWhenItsRequesterEndsIt(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		launcher := &fakeLauncher{}
		supervisor := testcheck.Must(New(launcher, 0))(t)
		defer supervisor.Close()
		answering := testcheck.Must(supervisor.Acquire(t.Context(), Servable{Name: "alpha"}))(t)
		swapped := make(chan *Lease, 1)
		go func() {
			lease, err := supervisor.Acquire(t.Context(), Servable{Name: "beta"})
			if err != nil {
				t.Error(err)
			}
			swapped <- lease
		}()
		time.Sleep(time.Hour)
		synctest.Wait()
		launcher.mu.Lock()
		alpha := launcher.processes["alpha"]
		launcher.mu.Unlock()
		select {
		case lease := <-swapped:
			t.Fatalf("an hour in, the swap stopped alpha under a request still answering and went to %+v", lease)
		default:
		}
		if alpha.stopped.Load() {
			t.Fatal("an hour in, alpha was stopped under a request still answering")
		}
		answering.Release()
		lease := <-swapped
		if lease == nil || lease.Target != "http://upstream/beta" || !alpha.stopped.Load() {
			t.Fatalf("once its requester ended the request, the swap went to %+v, alpha stopped %t", lease, alpha.stopped.Load())
		}
		lease.Release()
	})
}
