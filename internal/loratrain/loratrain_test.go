package loratrain

import (
	"encoding/json"
	"maps"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/densecausal"
	"overgo/internal/optimizer"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// projectAll projects every module's gradient from grads, as a step does for the modules
// the layers settle and the pass returns.
func projectAll(trainer *Trainer, grads densecausal.Grads) error {
	for name := range trainer.adapter.Modules {
		if err := trainer.project(name, grads); err != nil {
			return err
		}
	}
	return nil
}

// embeddingScale makes the tiny model's frozen tied embedding large enough for
// its logits to separate: with it, Muon at a rate of 0.02 takes the loss on a
// repeated pattern from about four to a quarter in forty updates.
const embeddingScale = 30

// tinyModel is a two-layer dense causal model with two query heads, one key
// head and a vocabulary of eight, saved as a safetensors checkpoint and loaded
// back, so the adapter trains over the model a checkpoint directory loads as.
func tinyModel(t *testing.T) (model *densecausal.Model, directory string, original map[string][]float32) {
	t.Helper()
	directory = t.TempDir()
	spec := testutil.DenseCausalSpec{Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 2, Seed: 5}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	// The fixture's weights are N(0, 0.02), which keeps a frozen tied embedding's logits within a
	// few hundredths of uniform: no adapter could lower the loss below ln 8. A larger embedding
	// gives the logits room, as a trained model's has.
	for index := range weights["model.embed_tokens.weight"] {
		weights["model.embed_tokens.weight"][index] *= embeddingScale
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config, err := json.Marshal(map[string]any{
		"model_type": "llama", "num_attention_heads": spec.Heads, "head_dim": spec.HeadDim, "max_position_embeddings": 16,
		"rope_theta": 10000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "config.json"), config, 0o600); err != nil {
		t.Fatal(err)
	}
	model, err = densecausal.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	original = make(map[string][]float32, len(model.Weights))
	for name, values := range model.Weights {
		original[name] = slices.Clone(values)
	}
	return model, directory, original
}

var periodic = []int{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4}

func loss(t *testing.T, model *densecausal.Model, tokens []int) float64 {
	t.Helper()
	value, _, _, err := model.LossAndGrads(tokens)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// TestFactorGradientsMatchFiniteDifferences: the gradient this package projects
// onto each factor from the backward pass is the derivative of the adapted
// model's loss with respect to that factor, checked entry by entry against
// central differences of the loss, with the up factors made nonzero so both
// factors' gradients are nonzero.
func TestFactorGradientsMatchFiniteDifferences(t *testing.T) {
	checkFactorGradients(t, Config{Rank: 2, Alpha: 6, Seed: 1}, 1e-2)
}

// checkFactorGradients is that check for the adapter the config starts, with central differences taken the given step apart: a larger scale bends the loss more, so needs a smaller one.
func checkFactorGradients(t *testing.T, config Config, step float64) {
	t.Helper()
	model, _, _ := tinyModel(t)
	trainer, err := New(model, config)
	if err != nil {
		t.Fatal(err)
	}
	random := rand.New(rand.NewSource(2))
	// In name order, so the same adapter is checked on every run: a map's order would draw another.
	for _, name := range slices.Sorted(maps.Keys(trainer.adapter.Modules)) {
		up := trainer.adapter.Modules[name].Up
		for index := range up {
			up[index] = float32(random.NormFloat64()) / 4
		}
	}
	if err := trainer.apply(); err != nil {
		t.Fatal(err)
	}
	_, _, grads, err := model.LossAndGrads(periodic)
	if err != nil {
		t.Fatal(err)
	}
	if err := projectAll(trainer, grads); err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, name := range []string{"model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.k_proj", "model.layers.0.mlp.down_proj"} {
		module := trainer.adapter.Modules[name]
		for factorName, factor := range map[string][]float32{name + downFactor: module.Down, name + upFactor: module.Up} {
			for _, index := range []int{0, 3, len(factor) - 1} {
				original := factor[index]
				factor[index] = original + float32(step)
				if err := trainer.apply(); err != nil {
					t.Fatal(err)
				}
				up := loss(t, model, periodic)
				factor[index] = original - float32(step)
				if err := trainer.apply(); err != nil {
					t.Fatal(err)
				}
				down := loss(t, model, periodic)
				factor[index] = original
				numeric := (up - down) / (2 * step)
				analytic := float64(trainer.gradients[factorName][index])
				if math.Abs(analytic-numeric) > 2e-4+0.01*math.Abs(numeric) {
					t.Errorf("%s[%d]: gradient %v, central difference %v", factorName, index, analytic, numeric)
				}
				checked++
			}
		}
	}
	if checked != 18 {
		t.Fatalf("checked %d entries, want 18", checked)
	}
	if err := trainer.apply(); err != nil {
		t.Fatal(err)
	}
}

// TestTrainingLowersTheLossAndFreezesTheBase: Muon over the factors lowers the
// loss on a repeated pattern; the model's adapted weights are the frozen base
// plus the adapter's update at every point, every other tensor never moves, and
// the base comes back exactly.
func TestTrainingLowersTheLossAndFreezesTheBase(t *testing.T) {
	model, _, original := tinyModel(t)
	trainer, err := New(model, Config{Rank: 4, Alpha: 8, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(trainer.adapter.Modules) != 14 {
		t.Fatalf("%d modules, want seven projections in each of two layers", len(trainer.adapter.Modules))
	}
	// B starts at zero: the adapted model equals the base.
	for name, values := range original {
		if !slices.Equal(model.Weights[name], values) {
			t.Fatalf("%s differs from the base before any update", name)
		}
	}
	batches := make([][]int, 40)
	for index := range batches {
		batches[index] = periodic
	}
	var seen []int
	losses, err := trainer.Train(batches, nil, 0.02, 0.9, func(step int, _ float64, _ time.Duration) error { seen = append(seen, step); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(losses) != 40 || len(seen) != 40 || seen[39] != 39 {
		t.Fatalf("%d losses and %d observed steps, want forty", len(losses), len(seen))
	}
	if first, last := losses[0], losses[len(losses)-1]; !(last < 0.5*first) {
		t.Fatalf("loss went from %v to %v, want a clear fall", first, last)
	}
	adapted := map[string]bool{}
	for name := range trainer.adapter.Modules {
		adapted[name+".weight"] = true
		delta, err := trainer.adapter.Delta(name)
		if err != nil {
			t.Fatal(err)
		}
		for index, value := range model.Weights[name+".weight"] {
			if want := original[name+".weight"][index] + delta[index]; math.Abs(float64(value-want)) > 1e-6 {
				t.Fatalf("%s[%d] = %v, want the base plus the update %v", name, index, value, want)
			}
		}
	}
	for name, values := range original {
		if !adapted[name] && !slices.Equal(model.Weights[name], values) {
			t.Fatalf("%s is not adapted yet changed", name)
		}
	}
	moved := false
	for name := range adapted {
		moved = moved || !slices.Equal(model.Weights[name], original[name])
	}
	if !moved {
		t.Fatal("training changed no adapted weight")
	}
	trainer.RestoreBase()
	for name, values := range original {
		if !slices.Equal(model.Weights[name], values) {
			t.Fatalf("%s is not the base after RestoreBase", name)
		}
	}
	if state, err := trainer.OptimizerState(); err != nil || state.Step != 40 || state.Parameters() != trainer.Plan().ParameterCount() {
		t.Fatalf("optimizer state = %+v (%v), want forty steps over the factors' parameters", state, err)
	}
}

// TestAdapterCarriesTheTrainingThroughPEFT: the trained adapter saves as PEFT
// and loads back; resuming it on a fresh model gives the trained model's
// weights; and merging it into the base checkpoint gives the same weights, so
// the file means exactly what the trainer trained.
func TestAdapterCarriesTheTrainingThroughPEFT(t *testing.T) {
	model, directory, _ := tinyModel(t)
	trainer, err := New(model, Config{Rank: 2, Alpha: 4, Seed: 3, BaseModel: "tiny"})
	if err != nil {
		t.Fatal(err)
	}
	batches := [][]int{periodic, periodic, periodic, periodic, periodic, periodic}
	if _, err := trainer.Train(batches, nil, 0.02, 0.9, nil); err != nil {
		t.Fatal(err)
	}
	saved := filepath.Join(t.TempDir(), "adapter")
	if err := peft.Save(saved, trainer.Adapter()); err != nil {
		t.Fatal(err)
	}
	loaded, err := peft.Load(saved)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Config.BaseModel != "tiny" || len(loaded.Modules) != len(trainer.adapter.Modules) {
		t.Fatalf("loaded %+v with %d modules", loaded.Config, len(loaded.Modules))
	}
	fresh, err := densecausal.Load(directory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Resume(fresh, loaded); err != nil {
		t.Fatal(err)
	}
	merged := filepath.Join(t.TempDir(), "merged")
	if _, err := peft.Merge(loaded, directory, merged); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(merged)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	for name := range loaded.Modules {
		tensor := name + ".weight"
		fromFile, err := safetensors.ReadF32(source.Tensors[tensor])
		if err != nil {
			t.Fatal(err)
		}
		for index := range fromFile {
			trained, resumed := model.Weights[tensor][index], fresh.Weights[tensor][index]
			if math.Abs(float64(trained-resumed)) > 1e-6 || math.Abs(float64(trained-fromFile[index])) > 1e-5 {
				t.Fatalf("%s[%d]: trained %v, resumed %v, merged on disk %v", tensor, index, trained, resumed, fromFile[index])
			}
		}
	}
}

// TestRefusesWhatItCannotTrain: a rank the projection cannot hold, a model with
// no target projection, a rate that changes after the first update, an adapter
// whose shapes are not the model's, and no batches.
func TestRefusesWhatItCannotTrain(t *testing.T) {
	model, _, _ := tinyModel(t)
	for name, scenario := range map[string]struct {
		config Config
		want   string
	}{
		"rank too large": {Config{Rank: 9, Alpha: 8}, "exceeds"},
		"no alpha":       {Config{Rank: 2}, "positive rank and alpha"},
		"no target":      {Config{Rank: 2, Alpha: 4, Targets: []string{"nothing_proj"}}, "no dense layer"},
	} {
		if _, err := New(model, scenario.config); err == nil || !strings.Contains(err.Error(), scenario.want) {
			t.Errorf("%s: err = %v, want it to say %q", name, err, scenario.want)
		}
	}
	trainer, err := New(model, Config{Rank: 2, Alpha: 4, Targets: []string{"q_proj"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(trainer.adapter.Modules) != 2 {
		t.Fatalf("%d modules for q_proj alone, want one per layer", len(trainer.adapter.Modules))
	}
	if _, err := trainer.Step(periodic, nil, 0.02, 0.9); err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.Step(periodic, nil, 0.5, 0.9); err == nil || !strings.Contains(err.Error(), "fixed by the first update") {
		t.Errorf("a changed rate: err = %v", err)
	}
	if _, err := trainer.Train(nil, nil, 0.02, 0.9, nil); err == nil {
		t.Error("training on no batches was accepted")
	}
	wrong := peft.Adapter{Config: peft.Config{Rank: 1, Alpha: 1}, Modules: map[string]peft.Module{
		"model.layers.0.self_attn.q_proj": {Rank: 1, In: 3, Out: 8, Alpha: 1, Down: make([]float32, 3), Up: make([]float32, 8)},
	}}
	if _, err := Resume(model, wrong); err == nil || !strings.Contains(err.Error(), "the model's") {
		t.Errorf("an adapter of another shape: err = %v", err)
	}
}

// trainedAdapter is the adapter a trainer holds after taking each batch once,
// with the flags for the tokens it learns, and the losses it saw.
func trainedAdapter(t *testing.T, batches [][]int, trained [][]bool) (peft.Adapter, []float64) {
	t.Helper()
	model, _, _ := tinyModel(t)
	trainer, err := New(model, Config{Rank: 2, Alpha: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	losses, err := trainer.Train(batches, trained, 0.02, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	return trainer.Adapter(), losses
}

// sameAdapter reports whether two adapters hold the same factors.
func sameAdapter(left, right peft.Adapter) bool {
	for name, module := range left.Modules {
		if !slices.Equal(module.Up, right.Modules[name].Up) || !slices.Equal(module.Down, right.Modules[name].Down) {
			return false
		}
	}
	return true
}

// TestMaskedBatchesLearnOnlyTheirTrainedTokens: an update's loss and factors do
// not depend on a token the batch does not train and nothing later reads, and
// do depend on it once it is trained.
func TestMaskedBatchesLearnOnlyTheirTrainedTokens(t *testing.T) {
	swapped := slices.Clone(periodic)
	swapped[len(swapped)-1] = 7
	batches := [][]int{periodic, periodic, periodic}
	others := [][]int{swapped, swapped, swapped}
	masked := make([]bool, len(periodic))
	for index := range masked {
		masked[index] = index < len(periodic)-1
	}
	learned := make([]bool, len(periodic))
	for index := range learned {
		learned[index] = true
	}
	flags := func(flag []bool) [][]bool { return [][]bool{flag, flag, flag} }

	adapter, losses := trainedAdapter(t, batches, flags(masked))
	otherAdapter, otherLosses := trainedAdapter(t, others, flags(masked))
	if !slices.Equal(losses, otherLosses) || !sameAdapter(adapter, otherAdapter) {
		t.Fatalf("a masked final token changed the run: losses %v against %v", losses, otherLosses)
	}
	adapter, losses = trainedAdapter(t, batches, flags(learned))
	otherAdapter, otherLosses = trainedAdapter(t, others, flags(learned))
	if slices.Equal(losses, otherLosses) || sameAdapter(adapter, otherAdapter) {
		t.Fatal("a trained final token changed nothing: the swap proves nothing")
	}
	model, _, _ := tinyModel(t)
	trainer, err := New(model, Config{Rank: 2, Alpha: 4})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.Train(batches, flags(masked)[:2], 0.02, 0.9, nil); err == nil {
		t.Error("flags for two of three batches were accepted")
	}
}

// TestAveragedTrainingEqualsTheAverageOfItsSteps trains one adapter six updates
// with a moving average of its factors and another identical one the same six,
// taking each update's factors: the first ends holding the average the recurrence
// gives them, written out here, and the model it is applied to holds the base plus
// that averaged adapter's update.
func TestAveragedTrainingEqualsTheAverageOfItsSteps(t *testing.T) {
	const (
		decay = 0.99
		steps = 6
	)
	batches := make([][]int, steps)
	for index := range batches {
		batches[index] = periodic
	}
	config := Config{Rank: 2, Alpha: 4, Seed: 1}

	model, _, _ := tinyModel(t)
	stepped, err := New(model, config)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string][2][]float64{}
	for name, module := range stepped.Adapter().Modules {
		expected[name] = [2][]float64{widen(module.Down), widen(module.Up)}
	}
	for update := 1; update <= steps; update++ {
		if _, err := stepped.Step(periodic, nil, 0.02, 0.9); err != nil {
			t.Fatal(err)
		}
		keep := min(decay, float64(1+update)/float64(10+update))
		for name, module := range stepped.Adapter().Modules {
			for factor, values := range [2][]float32{module.Down, module.Up} {
				for index, value := range values {
					expected[name][factor][index] = keep*expected[name][factor][index] + (1-keep)*float64(value)
				}
			}
		}
	}

	averagedModel, _, original := tinyModel(t)
	averaged, err := New(averagedModel, config)
	if err != nil {
		t.Fatal(err)
	}
	if err := averaged.Average(decay); err != nil {
		t.Fatal(err)
	}
	if _, err := averaged.Train(batches, nil, 0.02, 0.9, nil); err != nil {
		t.Fatal(err)
	}
	if err := averaged.UseAverage(); err != nil {
		t.Fatal(err)
	}
	for name, module := range averaged.Adapter().Modules {
		for factor, values := range [2][]float32{module.Down, module.Up} {
			for index, value := range values {
				if got, want := float64(value), expected[name][factor][index]; math.Abs(got-want) > 1e-5 {
					t.Fatalf("%s factor %d [%d] = %v, the average of the steps is %v", name, factor, index, got, want)
				}
			}
		}
	}
	for name, module := range averaged.Adapter().Modules {
		delta, err := averaged.Adapter().Delta(name)
		if err != nil {
			t.Fatal(err)
		}
		weight := averagedModel.Weights[name+weightSuffix]
		for index, value := range weight {
			if want := original[name+weightSuffix][index] + delta[index]; math.Abs(float64(value-want)) > 1e-5 {
				t.Fatalf("%s[%d] = %v, the base plus the averaged adapter is %v (module of rank %d)", name, index, value, want, module.Rank)
			}
		}
	}
	if err := stepped.UseAverage(); err == nil {
		t.Error("an average that was never started was used")
	}
	if err := stepped.Average(1); err == nil {
		t.Error("a decay of one was accepted")
	}
}

// TestAveragedTrainingResumesFromACheckpoint trains an adapter eight updates
// under an average, and again stopping after the fourth: a trainer over the
// adapter as it stood then, given the optimizer's state and the average, takes the
// last four and ends holding exactly the factors, and so the model, the
// uninterrupted run did.
func TestAveragedTrainingResumesFromACheckpoint(t *testing.T) {
	const (
		decay = 0.9
		stop  = 4
		steps = 8
	)
	batches := make([][]int, steps)
	for index := range batches {
		batches[index] = periodic
	}
	config := Config{Rank: 2, Alpha: 4, Seed: 1}

	wholeModel, _, _ := tinyModel(t)
	whole, err := New(wholeModel, config)
	if err != nil {
		t.Fatal(err)
	}
	if err := whole.Average(decay); err != nil {
		t.Fatal(err)
	}
	if _, err := whole.Train(batches, nil, 0.02, 0.9, nil); err != nil {
		t.Fatal(err)
	}
	if err := whole.UseAverage(); err != nil {
		t.Fatal(err)
	}

	firstModel, _, _ := tinyModel(t)
	first, err := New(firstModel, config)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Average(decay); err != nil {
		t.Fatal(err)
	}
	if _, err := first.Train(batches[:stop], nil, 0.02, 0.9, nil); err != nil {
		t.Fatal(err)
	}
	state, err := first.OptimizerState()
	if err != nil {
		t.Fatal(err)
	}
	average, err := optimizer.RestoreAverage(slices.Clone(first.AverageState().Values()), first.AverageState().Updates(), first.AverageState().Decay())
	if err != nil {
		t.Fatal(err)
	}
	// What a checkpoint holds: the adapter's factors, copied, and the model its factors adapt.
	stopped := peft.Adapter{Config: first.Adapter().Config, Modules: map[string]peft.Module{}}
	for name, module := range first.Adapter().Modules {
		module.Down, module.Up = slices.Clone(module.Down), slices.Clone(module.Up)
		stopped.Modules[name] = module
	}

	resumedModel, _, _ := tinyModel(t)
	resumed, err := Resume(resumedModel, stopped)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.Restore(state, average, 0.02, 0.9); err != nil {
		t.Fatal(err)
	}
	if _, err := resumed.Train(batches[stop:], nil, 0.02, 0.9, nil); err != nil {
		t.Fatal(err)
	}
	if err := resumed.UseAverage(); err != nil {
		t.Fatal(err)
	}
	for name, module := range whole.Adapter().Modules {
		other := resumed.Adapter().Modules[name]
		if !slices.Equal(module.Down, other.Down) || !slices.Equal(module.Up, other.Up) {
			t.Fatalf("%s differs between the uninterrupted run and the one resumed from its fourth update", name)
		}
	}
	for name, values := range wholeModel.Weights {
		if !slices.Equal(values, resumedModel.Weights[name]) {
			t.Fatalf("model tensor %s differs between the uninterrupted and the resumed run", name)
		}
	}
	if err := resumed.Restore(state, nil, 0.5, 0.9); err == nil || !strings.Contains(err.Error(), "fixed by the first update") {
		t.Errorf("a changed rate on restore: err = %v", err)
	}
}

func widen(values []float32) []float64 {
	wide := make([]float64, len(values))
	for index, value := range values {
		wide[index] = float64(value)
	}
	return wide
}
