package densecausal

import (
	"context"

	"overgo/internal/graphruntime"
	"overgo/internal/tensor"
)

// NewBlockAdapter lets the test outside the package build a block adapter:
// it sets one beside the trainers that import this package, which a test
// inside it cannot name.
var NewBlockAdapter = newBlockAdapter

// Step is the adapter's step, for that test.
func (adapter *blockAdapter) Step(state, target []float32, learningRate, momentum float64) (float64, error) {
	return adapter.step(state, target, learningRate, momentum)
}

// Passes is what a step executes, its forward and backward, with no update
// taken, for that test.
func (adapter *blockAdapter) Passes(state, target []float32) error {
	_, err := adapter.run(adapter.outputs, map[*tensor.Tensor][]float32{adapter.state: state, adapter.target: target})
	return err
}

// OnDevice is the adapter's move to a device, for that test.
func (adapter *blockAdapter) OnDevice(ctx context.Context, ordinal int) (*graphruntime.ResidentSession, error) {
	return adapter.onDevice(ctx, ordinal)
}
