package loop

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

type progressWorld struct {
	fakeWorld
	accepted []bool
	proofErr error
}

func (w *progressWorld) ProgressCheckpoint() (string, error) {
	return fmt.Sprint(w.call), nil
}

func (w *progressWorld) AcceptedProgress(_ Step, checkpoint string) (bool, error) {
	if checkpoint != fmt.Sprint(w.call-1) {
		return false, errors.New("checkpoint did not precede this worker")
	}
	return w.call <= len(w.accepted) && w.accepted[w.call-1], w.proofErr
}

func TestLoopAcceptedPrerequisiteProgress(t *testing.T) {
	world := &progressWorld{
		fakeWorld: fakeWorld{steps: []Step{{Item: "grammar", ID: "do"}}, failure: "hash-once remains open"},
		accepted:  []bool{false, true, true, false, false},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 9})
	if err != nil || outcome.Reason != ReasonParked || outcome.Invocations != 5 || len(outcome.Parked) != 1 {
		t.Fatalf("useful prerequisites followed by genuine nonprogress: %+v %v", outcome, err)
	}
	if len(world.steps) != 1 || !strings.Contains(world.feedback[3], "remaining acceptance is still open") {
		t.Fatal("prerequisite completion incorrectly completed the parent or lost its remaining acceptance")
	}
	if !strings.Contains(world.parked[0], "hash-once remains open") {
		t.Fatal("parking lost the actual outstanding failure")
	}
}

func TestLoopPrerequisiteProgressPreservesInvocationBudget(t *testing.T) {
	world := &progressWorld{
		fakeWorld: fakeWorld{steps: []Step{{Item: "grammar", ID: "do"}}},
		accepted:  []bool{true, true, true},
	}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 3})
	if err != nil || outcome.Reason != ReasonBudget || outcome.Invocations != 3 || len(outcome.Parked) != 0 || len(world.steps) != 1 {
		t.Fatalf("progress renewed total allowance: %+v %v", outcome, err)
	}
}

func TestLoopUnacceptedProgressStillParks(t *testing.T) {
	for _, name := range []string{"unchanged receipt", "replan", "merge", "foreign receipt", "pending validation", "failed validation", "green verifier", "zero-match verification"} {
		t.Run(name, func(t *testing.T) {
			world := &progressWorld{fakeWorld: fakeWorld{steps: []Step{{Item: "grammar", ID: "do"}}}}
			if name != "green verifier" {
				world.failure = name
			}
			outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
			if err != nil || outcome.Reason != ReasonParked || outcome.Invocations != 2 {
				t.Fatalf("unaccepted work earned progress: %+v %v", outcome, err)
			}
		})
	}
}

func TestLoopInvalidProgressProofRefusesCredit(t *testing.T) {
	invalid := errors.New("invalid completion proof")
	world := &progressWorld{fakeWorld: fakeWorld{steps: []Step{{Item: "grammar", ID: "do"}}}, accepted: []bool{true}, proofErr: invalid}
	outcome, err := Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	if !errors.Is(err, invalid) || outcome.Invocations != 1 || len(outcome.Parked) != 0 {
		t.Fatalf("invalid proof accepted: %+v %v", outcome, err)
	}
}
