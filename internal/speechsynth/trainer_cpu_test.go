package speechsynth

import (
	"maps"
	"math"
	"math/rand"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/optimizer"
)

func cpuJointFixture(t *testing.T) (*Model, *JointTrainer, TrainingExample) {
	t.Helper()
	rng := rand.New(rand.NewSource(31))
	model := tinyJointModel(rng)
	trainer, err := NewJointTrainer(model, optimizer.Config{Steps: 3, BaseLearningRate: 1e-3, Momentum: 0.95, Schedule: optimizer.ScheduleLinearDecay}, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = trainer.Close() })
	return model, trainer, TrainingExample{TextIDs: []int{2, 4, 1}, Frames: 2, Seed: 91, Latents: normVec(rng, 2*model.Dims.LatentDim, 0, 1)}
}

func jointWeightSnapshot(model *Model) map[string][]float32 {
	weights, _ := model.TrainedTensors(true)
	for name, values := range weights {
		weights[name] = slices.Clone(values)
	}
	return weights
}

func TestCPUJointTrainerExactResume(t *testing.T) {
	model, trainer, example := cpuJointFixture(t)
	initial := jointWeightSnapshot(model)
	frozenText, frozenInput := slices.Clone(model.condEmbed), slices.Clone(model.inputLinear)
	firstLoss, err := trainer.Step(example)
	if err != nil {
		t.Fatal(err)
	}
	after, err := model.LossAndGrads(example.TextIDs, example.Latents, example.Frames, example.Seed, nil)
	if err != nil || !(after < firstLoss) || maps.EqualFunc(initial, jointWeightSnapshot(model), slices.Equal[[]float32]) {
		t.Fatalf("CPU update: loss %g -> %g, error=%v", firstLoss, after, err)
	}
	weights, state := jointWeightSnapshot(model), mustJointState(t, trainer)
	addresses := map[string]*float32{}
	for name, values := range trainer.gradients {
		addresses[name] = &values[0]
	}
	var wantLosses []float64
	for range 2 {
		example.Seed++
		loss, err := trainer.Step(example)
		if err != nil {
			t.Fatal(err)
		}
		wantLosses = append(wantLosses, loss)
	}
	for name, values := range trainer.gradients {
		if addresses[name] != &values[0] {
			t.Fatalf("gradient storage replaced for %s", name)
		}
	}
	if !slices.Equal(frozenText, model.condEmbed) || !slices.Equal(frozenInput, model.inputLinear) {
		t.Fatal("frozen input tensors changed")
	}
	resumedModel, resumed, next := cpuJointFixture(t)
	if err := resumed.Restore(weights, state); err != nil {
		t.Fatal(err)
	}
	for index := range 2 {
		next.Seed++
		loss, err := resumed.Step(next)
		if err != nil || loss != wantLosses[index] {
			t.Fatalf("resumed step %d loss=%g want=%g: %v", index, loss, wantLosses[index], err)
		}
	}
	if !maps.EqualFunc(jointWeightSnapshot(model), jointWeightSnapshot(resumedModel), slices.Equal[[]float32]) || !reflect.DeepEqual(mustJointState(t, trainer), mustJointState(t, resumed)) {
		t.Fatal("CPU weights or optimizer/scheduler state differ after resume")
	}
	if err := resumed.Close(); err != nil {
		t.Fatal(err)
	}
	if resumed.model != nil || resumed.weights != nil || resumed.gradients != nil {
		t.Fatal("closed trainer retains model or numeric buffers")
	}
	if err := resumed.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := resumed.Step(next); err == nil {
		t.Fatal("closed trainer stepped")
	}
	if _, err := resumed.OptimizerSnapshot(); err == nil {
		t.Fatal("closed trainer checkpointed")
	}
	if err := resumed.Restore(weights, state); err == nil {
		t.Fatal("closed trainer restored")
	}
}

func mustJointState(t *testing.T, trainer *JointTrainer) optimizer.State {
	t.Helper()
	state, err := trainer.OptimizerSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	return state
}

func TestCPUJointTrainerFailedUpdateRequiresRestore(t *testing.T) {
	model, _, example := cpuJointFixture(t)
	// A finite but unrepresentable F32 update exercises post-update refusal.
	trainer, err := NewJointTrainer(model, optimizer.Config{Steps: 1, BaseLearningRate: math.MaxFloat64, Momentum: 0.95}, true)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = trainer.Close() })
	weights, state := jointWeightSnapshot(model), mustJointState(t, trainer)
	if _, err := trainer.Step(example); err == nil {
		t.Fatal("non-finite update admitted")
	}
	if _, err := trainer.OptimizerSnapshot(); err == nil {
		t.Fatal("failed update exposed a checkpoint boundary")
	}
	if _, err := trainer.Step(example); err == nil {
		t.Fatal("failed update admitted another step")
	}
	if err := trainer.Restore(weights, state); err != nil {
		t.Fatal(err)
	}
	if !maps.EqualFunc(weights, jointWeightSnapshot(model), slices.Equal[[]float32]) || !reflect.DeepEqual(state, mustJointState(t, trainer)) {
		t.Fatal("restore did not recover the last valid boundary")
	}
}

func TestCPUJointTrainerRefusesInvalidRestoreAndTargets(t *testing.T) {
	model, trainer, example := cpuJointFixture(t)
	if _, err := trainer.Step(example); err != nil {
		t.Fatal(err)
	}
	wantWeights, wantState := jointWeightSnapshot(model), mustJointState(t, trainer)
	for _, failure := range []string{"configuration", "plan", "momentum", "tensor"} {
		t.Run(failure, func(t *testing.T) {
			weights, state := jointWeightSnapshot(model), mustJointState(t, trainer)
			switch failure {
			case "configuration":
				state.Config.BaseLearningRate *= 2
			case "plan":
				state.PlanIdentity = "different"
			case "momentum":
				state.Momentum[len(state.Momentum)-1] = math.NaN()
			case "tensor":
				for name := range weights {
					delete(weights, name)
					break
				}
			}
			if err := trainer.Restore(weights, state); err == nil {
				t.Fatal("invalid resume admitted")
			}
			if !maps.EqualFunc(wantWeights, jointWeightSnapshot(model), slices.Equal[[]float32]) || !reflect.DeepEqual(wantState, mustJointState(t, trainer)) {
				t.Fatal("rejected restore mutated live state")
			}
		})
	}
	example.Latents[0] = float32(math.NaN())
	if _, err := trainer.Step(example); err == nil {
		t.Fatal("non-finite target admitted")
	}
	if !maps.EqualFunc(wantWeights, jointWeightSnapshot(model), slices.Equal[[]float32]) || !reflect.DeepEqual(wantState, mustJointState(t, trainer)) {
		t.Fatal("rejected target changed checkpoint state")
	}
}
