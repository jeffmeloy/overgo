package loop

import "testing"

type proposalFakeWorld struct {
	fakeWorld
	facts     ClosureFacts
	proposals int
}

func (world *proposalFakeWorld) ClosureFacts() (ClosureFacts, error) { return world.facts, nil }
func (world *proposalFakeWorld) AdmitNextProposal() (bool, error) {
	if world.proposals != 0 {
		return false, nil
	}
	world.proposals++
	world.steps = []Step{{Item: "steered", ID: "do"}}
	world.behavior = []func(*fakeWorld, Step, string) string{func(w *fakeWorld, _ Step, _ string) string { w.steps = nil; return "" }}
	return true, nil
}

func TestProposalClosure(t *testing.T) {
	closure := &ClosureConfig{MaxWallNS: 100, MaxCostUnits: 100, MaxMutations: 10, MaxExperiments: 2, MaxProposals: 2}
	world := &proposalFakeWorld{facts: ClosureFacts{MeasuredGain: true}}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 4, Closure: closure})
	if err != nil {
		t.Fatal(err)
	}
	if outcome.Reason != ReasonPlanComplete || outcome.Invocations != 1 || world.proposals != 1 {
		t.Fatalf("closure = %+v proposals=%d", outcome, world.proposals)
	}
	blocked := &proposalFakeWorld{facts: ClosureFacts{MeasuredGain: true, OutstandingObligations: 1}}
	outcome, err = Run(blocked, Config{MaxAttemptsPerStep: 2, MaxInvocations: 4, Closure: closure})
	if err != nil || outcome.Reason != ReasonClosureBlocked {
		t.Fatalf("blocked = %+v %v", outcome, err)
	}
	stopped := &proposalFakeWorld{facts: ClosureFacts{MeasuredGain: true, OperatorStop: true}}
	outcome, err = Run(stopped, Config{MaxAttemptsPerStep: 2, MaxInvocations: 4, Closure: closure})
	if err != nil || outcome.Reason != ReasonOperatorStop {
		t.Fatalf("stopped = %+v %v", outcome, err)
	}
}
