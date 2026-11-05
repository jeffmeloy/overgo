//go:build windows

package scratchtrain

import (
	"math"
	"testing"
	"time"

	"overgo/internal/checkpointtest"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/scratchmodel"
	"overgo/internal/scratchmodeltest"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

func TestScratchResidentTrajectoryParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, scratchmodeltest.Profile(t))
	testcheck.NoError(t, err)
	const steps = 3
	policy := trainingprogram.BuiltinOptimizerPolicy()
	host, err := construction.TrainShared(steps, policy)
	testcheck.NoError(t, err)
	initializedAt := time.Now()
	trainer, err := NewResidentTrainer(construction, steps, policy)
	testcheck.NoError(t, err)
	defer trainer.Close()
	initWall := time.Since(initializedAt)
	// A resident cache hit must enforce the same boundary as a fresh graph.
	overflow := make([]int, construction.Config().BlockSize+2)
	invalidTail := make([]int, construction.Config().BlockSize+1)
	invalidTail[len(invalidTail)-1] = construction.Config().VocabSize
	for _, tokens := range [][]int{overflow, invalidTail} {
		if _, err := trainer.forwardProgram(tokens); err == nil {
			t.Fatal("resident program cache bypassed input admission")
		}
	}
	losses, walls := make([]float64, steps), make([]time.Duration, steps)
	for step := range steps {
		tokens, err := construction.Tokens(construction.Split().Train[step%len(construction.Split().Train)])
		testcheck.NoError(t, err)
		started := time.Now()
		losses[step], err = trainer.Step(tokens, step+1)
		walls[step] = time.Since(started)
		testcheck.NoError(t, err)
	}
	validationTokens, err := construction.Tokens(construction.Split().Validation[0])
	testcheck.NoError(t, err)
	validation, err := trainer.Evaluate(validationTokens)
	testcheck.NoError(t, err)
	worst := math.Abs(validation - host.ValidationLoss)
	for step := range steps {
		worst = max(worst, math.Abs(losses[step]-host.Losses[step]))
	}
	weights, gradients, momentum, err := trainer.Snapshot()
	testcheck.NoError(t, err)
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

// TestScratchResidentCheckpointRoundTrip trains the oracle construction on
// the device and holds what it trained to the checkpoint it publishes: the
// Muon state of the last step and every parameter's trained weights read
// back bit for bit under the construction's program.
func TestScratchResidentCheckpointRoundTrip(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, scratchmodeltest.Profile(t))
	testcheck.NoError(t, err)
	const steps = 2
	trainer, err := NewResidentTrainer(construction, steps, trainingprogram.BuiltinOptimizerPolicy())
	testcheck.NoError(t, err)
	defer trainer.Close()
	for step := range steps {
		tokens, err := construction.Tokens(construction.Split().Train[step%len(construction.Split().Train)])
		testcheck.NoError(t, err)
		if _, err := trainer.Step(tokens, step+1); err != nil {
			t.Fatal(err)
		}
	}
	trained, err := trainer.Trained()
	testcheck.NoError(t, err)
	if step := trained.Snapshot().Step; step != steps {
		t.Fatalf("trained state is at step %d, want the last step %d", step, steps)
	}
	weights, _, _, err := trainer.Snapshot()
	testcheck.NoError(t, err)
	groups := construction.OptimizerPlan().Groups()
	parameters := make(map[string][]float32, len(groups))
	for _, group := range groups {
		parameters[group.Name] = weights[group.Start:group.End]
	}
	checkpointtest.RequireRoundTrip(t, construction.Program(), trained, parameters)
}

func TestScratchResidentStateParity(t *testing.T) {
	cudatest.Require(t)
	oracle := loadOracle(t)
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: oracle.Documents, Seed: oracle.Seed, Steps: oracle.Steps}, scratchmodeltest.Profile(t))
	testcheck.NoError(t, err)
	const steps = 3
	hostWeights := construction.InitialWeights()
	hostGradients := make([]float32, len(hostWeights))
	hostModel, err := construction.BindMADTransformer(hostWeights)
	testcheck.NoError(t, err)
	hostGradient, err := construction.BindMADTransformer(hostGradients)
	testcheck.NoError(t, err)
	config, err := construction.OptimizerConfig(trainingprogram.BuiltinOptimizerPolicy(), steps)
	testcheck.NoError(t, err)
	hostMuon, err := optimizer.New(hostWeights, hostGradients, construction.OptimizerPlan(), config)
	testcheck.NoError(t, err)
	trainer, err := NewResidentTrainer(construction, steps, trainingprogram.BuiltinOptimizerPolicy())
	testcheck.NoError(t, err)
	defer trainer.Close()
	backward, err := trainer.training.Select(trainingprogram.PhaseBatch, trainingprogram.PhaseForward, trainingprogram.PhaseBackward)
	testcheck.NoError(t, err)
	optimize, err := trainer.training.Select(trainingprogram.PhaseOptimize)
	testcheck.NoError(t, err)

	for step := range steps {
		tokens, err := construction.Tokens(construction.Split().Train[step%len(construction.Split().Train)])
		testcheck.NoError(t, err)
		hostLoss, trace, err := hostmath.MADTransformerForward(hostModel, tokens)
		testcheck.NoError(t, err)
		testcheck.NoError(t, hostmath.MADTransformerBackward(hostModel, hostGradient, tokens, trace))
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
		testcheck.NoError(t, err)
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
