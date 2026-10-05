package loratrain

import (
	"slices"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/testutil"
)

// quantizedLayers is the depth of the fixture the test trains over: enough layers that the
// scratch one layer reads through is a small part of what a dense model would hold.
const quantizedLayers = 6

// TestAdapterOnQuantizedBase trains the same adapter over a Q8_0 GGUF held two ways,
// every weight dequantized at load and the adapted projections kept as stored and
// dequantized as each layer reads them: the two runs have the same losses and end
// with the same factors, to the last bit, since the weights the layers read are the
// same; the second model holds under half the weight bytes of the first; stopping
// the update returns it to the base; and a model with frozen weights is refused
// anything but an adapter over them.
func TestAdapterOnQuantizedBase(t *testing.T) {
	path := testutil.QuantizedDenseGGUF(t, quantizedLayers)
	batches := make([][]int, 6)
	for index := range batches {
		batches[index] = periodic
	}
	config := Config{Rank: 4, Alpha: 8, Seed: 1}

	dense, err := densecausal.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	frozen, err := densecausal.LoadGGUFFrozen(path, Adapts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if !frozen.Frozen("model.layers.0.self_attn.q_proj.weight") || !frozen.Frozen("model.layers.5.mlp.down_proj.weight") || frozen.Frozen("model.embed_tokens.weight") {
		t.Fatal("the model froze the wrong weights: want the seven projections of every layer, and nothing else")
	}
	denseFull, _, _ := dense.WeightBytes()
	frozenFull, frozenStored, frozenScratch := frozen.WeightBytes()
	t.Logf("weight bytes: dense %d, with frozen weights %d (%d full, %d stored, %d scratch)", denseFull, frozenFull+frozenStored+frozenScratch, frozenFull, frozenStored, frozenScratch)
	if resident := frozenFull + frozenStored + frozenScratch; resident*2 >= denseFull {
		t.Fatalf("the model with frozen weights holds %d bytes (%d full, %d stored, %d scratch) against the dense model's %d: want under half", resident, frozenFull, frozenStored, frozenScratch, denseFull)
	}
	base := loss(t, dense, periodic)
	if got := loss(t, frozen, periodic); got != base {
		t.Fatalf("before an adapter the frozen model's loss is %v, the dense model's %v", got, base)
	}

	denseTrainer, err := New(dense, config)
	if err != nil {
		t.Fatal(err)
	}
	frozenTrainer, err := New(frozen, config)
	if err != nil {
		t.Fatal(err)
	}
	denseLosses, err := denseTrainer.Train(batches, nil, 0.02, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	frozenLosses, err := frozenTrainer.Train(batches, nil, 0.02, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(denseLosses, frozenLosses) {
		t.Fatalf("losses differ: dense %v, frozen %v", denseLosses, frozenLosses)
	}
	if denseLosses[len(denseLosses)-1] >= denseLosses[0] {
		t.Fatalf("the adapter did not lower the loss: %v", denseLosses)
	}
	for name, module := range denseTrainer.Adapter().Modules {
		other := frozenTrainer.Adapter().Modules[name]
		if !slices.Equal(module.Down, other.Down) || !slices.Equal(module.Up, other.Up) {
			t.Fatalf("%s: the factors differ between the dense and the frozen run", name)
		}
	}
	if got, want := loss(t, frozen, periodic), loss(t, dense, periodic); got != want {
		t.Fatalf("after training the frozen model's loss is %v, the dense model's %v", got, want)
	}

	frozenTrainer.RestoreBase()
	if got := loss(t, frozen, periodic); got != base {
		t.Fatalf("with the update stopped the loss is %v, the base's %v", got, base)
	}
	if _, err := frozen.TrainingPlan(); err == nil {
		t.Error("a model with frozen weights was given a plan to train its own")
	}
	if _, err := densecausal.LoadGGUFFrozen(path, nil); err == nil {
		t.Error("freezing weights without naming them was accepted")
	}
}
