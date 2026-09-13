package loop

import (
	"fmt"
	"testing"
)

type waitingDispatchWorld struct{ fakeWorld }

func (w *waitingDispatchWorld) Current() (Step, bool, error) {
	return Step{}, false, fmt.Errorf("%w: claimed by another worker until release", ErrWorkWaiting)
}

func TestDispatchContentionPreservesWork(t *testing.T) {
	world := &waitingDispatchWorld{}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1})
	if err != nil || outcome.Reason != ReasonWorkWaiting || world.call != 0 || len(world.parked) != 0 {
		t.Fatalf("contention became a failure, execution or completion: %+v %v", outcome, err)
	}
}
