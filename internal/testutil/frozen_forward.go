package testutil

import (
	"context"
	"slices"

	"overgo/internal/artifact"
)

// FrozenForwardStub is the shared test double for bridge-training frozen
// forwards: it replays its input unchanged and counts invocations, so a
// test can prove exactly how often the frozen models executed.
type FrozenForwardStub struct {
	Model artifact.ID
	Calls int
}

// ModelID names the frozen model the stub stands in for.
func (forward *FrozenForwardStub) ModelID() artifact.ID { return forward.Model }

// Forward replays the input unchanged, counting the call.
func (forward *FrozenForwardStub) Forward(_ context.Context, input []float32) ([]float32, error) {
	forward.Calls++
	return slices.Clone(input), nil
}
