//go:build windows

package scratchmodel

import (
	"math"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/trainingprogram"
)

func TestScratchResidentTrajectoryParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, testDerivationProfile(t))
	if err != nil {
		t.Fatal(err)
	}
	const steps = 3
	host, err := construction.TrainShared(steps)
	if err != nil {
		t.Fatal(err)
	}
	initializedAt := time.Now()
	trainer, err := NewResidentTrainer(construction, steps)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	initWall := time.Since(initializedAt)
	losses, walls := make([]float64, steps), make([]time.Duration, steps)
	for step := range steps {
		tokens, err := construction.Tokens(construction.split.Train[step%len(construction.split.Train)])
		if err != nil {
			t.Fatal(err)
		}
		started := time.Now()
		losses[step], err = trainer.Step(tokens, step+1)
		walls[step] = time.Since(started)
		if err != nil {
			t.Fatal(err)
		}
	}
	validationTokens, err := construction.Tokens(construction.split.Validation[0])
	if err != nil {
		t.Fatal(err)
	}
	validation, err := trainer.Evaluate(validationTokens)
	if err != nil {
		t.Fatal(err)
	}
	worst := math.Abs(validation - host.ValidationLoss)
	for step := range steps {
		worst = max(worst, math.Abs(losses[step]-host.Losses[step]))
	}
	weights, gradients, momentum, err := trainer.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	for _, slab := range [][]float32{weights, gradients, momentum} {
		for _, value := range slab {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("resident scratch state is non-finite")
			}
		}
	}
	t.Logf("resident scratch init=%s cold=%s warm=%s/%s losses=%v validation=%.9f host max delta=%.3e", initWall, walls[0], walls[1], walls[2], losses, validation, worst)
	if worst > 3e-3 {
		t.Fatalf("resident scratch trajectory delta %.3e", worst)
	}
}

func TestScratchResidentStateParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := Compile(CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, testDerivationProfile(t))
	if err != nil {
		t.Fatal(err)
	}
	const steps = 3
	hostWeights := slices.Clone(construction.weights)
	hostGradients := make([]float32, len(hostWeights))
	hostModel, err := construction.bindMADTransformer(hostWeights)
	if err != nil {
		t.Fatal(err)
	}
	hostGradient, err := construction.bindMADTransformer(hostGradients)
	if err != nil {
		t.Fatal(err)
	}
	hostMuon, err := optimizer.New(hostWeights, hostGradients, construction.optimizer, optimizer.Config{
		BaseLearningRate: construction.config.BaseLR,
		Momentum:         construction.config.MuonMomentum,
		Steps:            steps,
		Schedule:         optimizer.ScheduleLinearDecay,
	})
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewResidentTrainer(construction, steps)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	backward, err := trainer.training.Select(trainingprogram.PhaseBatch, trainingprogram.PhaseForward, trainingprogram.PhaseBackward)
	if err != nil {
		t.Fatal(err)
	}
	optimize, err := trainer.training.Select(trainingprogram.PhaseOptimize)
	if err != nil {
		t.Fatal(err)
	}

	for step := range steps {
		tokens, err := construction.Tokens(construction.split.Train[step%len(construction.split.Train)])
		if err != nil {
			t.Fatal(err)
		}
		hostLoss, trace, err := hostmath.MADTransformerForward(hostModel, tokens)
		if err != nil {
			t.Fatal(err)
		}
		if err := hostmath.MADTransformerBackward(hostModel, hostGradient, tokens, trace); err != nil {
			t.Fatal(err)
		}
		state := residentTrainingState{trainer: trainer, tokens: tokens, step: step + 1}
		if err := backward.Run(&state); err != nil {
			state.release()
			t.Fatal(err)
		}
		_, residentGradients, _, err := trainer.Snapshot()
		if err != nil {
			state.release()
			t.Fatal(err)
		}
		gradientDelta := maxF32StateDelta(residentGradients, hostGradients)
		if err := optimize.Run(&state); err != nil {
			state.release()
			t.Fatal(err)
		}
		state.release()
		hostMuon.Step()
		residentWeights, residentClearedGradients, residentMomentum, err := trainer.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		hostState := hostMuon.Snapshot()
		momentumDelta := maxF64F32StateDelta(residentMomentum, hostState.Momentum)
		weightDelta := maxF32StateDelta(residentWeights, hostWeights)
		if maxF32StateDelta(residentClearedGradients, make([]float32, len(residentClearedGradients))) != 0 {
			t.Fatalf("step %d resident gradients not consumed", step+1)
		}
		t.Logf("step=%d loss=%.9f host=%.9f gradient=%.3e weight=%.3e momentum=%.3e", step+1, state.loss, hostLoss, gradientDelta, weightDelta, momentumDelta)
		if math.Abs(state.loss-hostLoss) > f32ScalarBound(hostLoss) ||
			gradientDelta > f32StateBound(hostGradients) ||
			weightDelta > f32StateBound(hostWeights) ||
			momentumDelta > f64StateAsF32Bound(hostState.Momentum) {
			t.Fatalf("step %d resident state differs", step+1)
		}
	}
}

func f32ScalarBound(reference float64) float64 {
	return math.Sqrt(float64(math.Nextafter32(1, 2)-1)) * max(1, math.Abs(reference))
}

func f32StateBound(reference []float32) float64 {
	scale := float64(1)
	for _, value := range reference {
		scale = max(scale, math.Abs(float64(value)))
	}
	return math.Sqrt(float64(math.Nextafter32(1, 2)-1)) * scale
}

func f64StateAsF32Bound(reference []float64) float64 {
	scale := float64(1)
	for _, value := range reference {
		scale = max(scale, math.Abs(value))
	}
	return math.Sqrt(float64(math.Nextafter32(1, 2)-1)) * scale
}

func maxF32StateDelta(left, right []float32) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	var worst float64
	for index := range left {
		worst = max(worst, math.Abs(float64(left[index]-right[index])))
	}
	return worst
}

func maxF64F32StateDelta(left []float32, right []float64) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	var worst float64
	for index := range left {
		worst = max(worst, math.Abs(float64(left[index])-right[index]))
	}
	return worst
}
