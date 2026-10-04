package loratrain

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/densecausal"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

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
	model, _, _ := tinyModel(t)
	trainer, err := New(model, Config{Rank: 2, Alpha: 6, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	random := rand.New(rand.NewSource(2))
	for _, module := range trainer.adapter.Modules {
		for index := range module.Up {
			module.Up[index] = float32(random.NormFloat64()) / 4
		}
	}
	if err := trainer.apply(); err != nil {
		t.Fatal(err)
	}
	_, _, grads, err := model.LossAndGrads(periodic)
	if err != nil {
		t.Fatal(err)
	}
	if err := trainer.project(grads); err != nil {
		t.Fatal(err)
	}
	const step = 1e-2
	checked := 0
	for _, name := range []string{"model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.k_proj", "model.layers.0.mlp.down_proj"} {
		module := trainer.adapter.Modules[name]
		for factorName, factor := range map[string][]float32{name + downFactor: module.Down, name + upFactor: module.Up} {
			for _, index := range []int{0, 3, len(factor) - 1} {
				original := factor[index]
				factor[index] = original + step
				if err := trainer.apply(); err != nil {
					t.Fatal(err)
				}
				up := loss(t, model, periodic)
				factor[index] = original - step
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
	losses, err := trainer.Train(batches, 0.02, 0.9, func(step int, _ float64, _ time.Duration) error { seen = append(seen, step); return nil })
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
	if _, err := trainer.Train(batches, 0.02, 0.9, nil); err != nil {
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
	if _, err := trainer.Step(periodic, 0.02, 0.9); err != nil {
		t.Fatal(err)
	}
	if _, err := trainer.Step(periodic, 0.5, 0.9); err == nil || !strings.Contains(err.Error(), "fixed by the first update") {
		t.Errorf("a changed rate: err = %v", err)
	}
	if _, err := trainer.Train(nil, 0.02, 0.9, nil); err == nil {
		t.Error("training on no batches was accepted")
	}
	wrong := peft.Adapter{Config: peft.Config{Rank: 1, Alpha: 1}, Modules: map[string]peft.Module{
		"model.layers.0.self_attn.q_proj": {Rank: 1, In: 3, Out: 8, Alpha: 1, Down: make([]float32, 3), Up: make([]float32, 8)},
	}}
	if _, err := Resume(model, wrong); err == nil || !strings.Contains(err.Error(), "the model's") {
		t.Errorf("an adapter of another shape: err = %v", err)
	}
}
