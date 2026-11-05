package main

import (
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/hostoptimizer"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

// TestTernaryMastersCheckpointRoundTripIsBitExact steps a master matrix once
// and holds the checkpoint it publishes to reading back bit for bit.
func TestTernaryMastersCheckpointRoundTripIsBitExact(t *testing.T) {
	plan, err := hostoptimizer.CompilePlan(testMatrix.End, []hostoptimizer.GroupSpec{testMatrix})
	testcheck.NoError(t, err)
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(testMatrix.End)
	testcheck.NoError(t, err)
	masters, gradients := []float32{0.5, -0.25, 0.75, -1}, []float32{0.1, -0.2, 0.3, -0.4}
	muon, err := hostoptimizer.New(masters, gradients, plan, config)
	testcheck.NoError(t, err)
	muon.Step()
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveLatentL2, nil, plan)
	testcheck.NoError(t, err)
	checkpointtest.RequireRoundTrip(t, program, ternaryMasters{masters: masters, plan: plan, stepper: hostStepper{muon}},
		map[string][]float32{testMatrix.Name: masters})
}

// hostStepper steps a host Muon optimizer as the platform stepper would.
type hostStepper struct{ *hostoptimizer.Optimizer }

func (s hostStepper) Step() error {
	s.Optimizer.Step()
	return nil
}

func (hostStepper) Close() error { return nil }
