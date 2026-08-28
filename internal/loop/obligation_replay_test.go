package loop

import "testing"

// obligationWorld extends the scripted world with a replay counter and
// a programmed due count per supervision round.
type obligationWorld struct {
	fakeWorld
	replays int
	due     []uint64
}

func (w *obligationWorld) ReplayObligations() (uint64, error) {
	index := w.replays
	w.replays++
	if index < len(w.due) {
		return w.due[index], nil
	}
	return 0, nil
}

// TestObligationCompletionReplay pins the unconditional replay hook:
// the driver replays durable obligations at the top of every
// supervision round -- rounds that dispatch work, rounds where the
// worker fires, and the final round where nothing is left -- and the
// outcome reports the last observed due count.
func TestObligationCompletionReplay(t *testing.T) {
	world := &obligationWorld{
		fakeWorld: fakeWorld{
			steps: []Step{{Item: "alpha", ID: "do"}},
			behavior: []func(w *fakeWorld, step Step, feedback string) string{
				func(w *fakeWorld, _ Step, _ string) string {
					w.steps = w.steps[1:] // worker committed through the gate
					return ""
				},
			},
		},
		due: []uint64{3, 2},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 4})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete {
		t.Fatalf("outcome = %+v", outcome)
	}
	// Round one dispatched the step; round two found the plan drained.
	// The replay ran unconditionally in both.
	if world.replays != 2 {
		t.Fatalf("replay ran %d times over 2 supervision rounds", world.replays)
	}
	if outcome.ObligationsDue != 2 {
		t.Fatalf("outcome reports %d due follow-ups, last replay said 2", outcome.ObligationsDue)
	}

	// A world without the extension runs exactly as before.
	plain := &fakeWorld{}
	outcome, err = Run(plain, Config{MaxAttemptsPerStep: 1, MaxInvocations: 1})
	if err != nil || outcome.Reason != ReasonPlanComplete || outcome.ObligationsDue != 0 {
		t.Fatalf("plain world outcome = %+v err=%v", outcome, err)
	}
}
