package landing

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestAStepThatEndsTheStepsTakesNoMoreTurns holds HoldTestTurnEach to ending at a step that answers ErrNoMoreSteps
// (SQA-438): of three steps whose second ends them, two take their turns and the third takes none and does not run,
// the end no failure. A run of the lanes that yielded had taken the card's turn and the test turn for each step after
// its yield, to run nothing.
func TestAStepThatEndsTheStepsTakesNoMoreTurns(t *testing.T) {
	turns, ran := 0, 0
	steps := []func() error{
		func() error { ran++; return nil },
		func() error { ran++; return ErrNoMoreSteps },
		func() error { ran++; return nil },
	}
	testcheck.NoError(t, HoldTestTurnEach("a fixture lane", func(work func() error) error { turns++; return work() }, steps))
	if turns != 2 || ran != 2 {
		t.Errorf("three steps whose second ends them took %d turns and ran %d; want two of each", turns, ran)
	}
}
