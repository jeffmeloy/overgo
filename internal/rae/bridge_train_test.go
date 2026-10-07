package rae

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
)

// TestBridgeTrainsAgainstTheDecodersPicture holds the bridge trainer to its
// loss: the gradient it steps on agrees with the change in the loss when the
// bridge is nudged; steps from a disturbed bridge bring the decoder's picture
// back toward the pictures a known bridge drew, while the decoder itself does
// not move; and the trained bridge is what it saves.
func TestBridgeTrainsAgainstTheDecodersPicture(t *testing.T) {
	t.Parallel()
	decoder := smallDecoder(3)
	const sourceWidth = 5
	rng := rand.New(rand.NewPCG(5, 5))
	random := func(count int, spread float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = spread * float32(rng.NormFloat64())
		}
		return values
	}
	known := EncoderBridge{Weight: random(decoder.latent*sourceWidth, 0.5), Bias: random(decoder.latent, 0.1)}
	tokens := decoder.grid * decoder.grid
	examples := make([]BridgeExample, 6)
	for index := range examples {
		source := random(tokens*sourceWidth, 1)
		mapped, err := known.Apply(source)
		if err != nil {
			t.Fatal(err)
		}
		picture, err := decoder.decodeTokens(mapped)
		if err != nil {
			t.Fatal(err)
		}
		examples[index] = BridgeExample{SourceTokens: source, Picture: picture}
	}
	disturbed := EncoderBridge{Weight: append([]float32(nil), known.Weight...), Bias: append([]float32(nil), known.Bias...)}
	for index, nudge := range random(len(disturbed.Weight), 0.05) {
		disturbed.Weight[index] += nudge
	}
	frozen := append([]float32(nil), decoder.predWeight...)
	config := optimizer.Config{BaseLearningRate: 2e-3, Momentum: 0.9, Schedule: optimizer.ScheduleConstant}
	trainer, err := NewBridgeTrainer(&BridgedAutoencoder{bridge: disturbed, decoder: decoder}, config)
	if err != nil {
		t.Fatal(err)
	}
	if trainer.TrainableParameterCount() != decoder.latent*(sourceWidth+1) || trainer.OptimizerPlan().GroupCount() != 2 {
		t.Fatalf("the trainer holds %d weights in %d groups", trainer.TrainableParameterCount(), trainer.OptimizerPlan().GroupCount())
	}
	// The gradient of one pass against finite differences along a random direction.
	if _, err := trainer.pass(t.Context(), examples, true); err != nil {
		t.Fatal(err)
	}
	gradient := append([]float32(nil), trainer.gradients...)
	along := random(len(gradient), 1)
	var predicted float64
	for index, value := range gradient {
		predicted += float64(value) * float64(along[index])
	}
	const nudge = 2e-3
	original := append([]float32(nil), trainer.weights...)
	losses := [2]float64{}
	for side, sign := range [2]float32{1, -1} {
		for index := range trainer.weights {
			trainer.weights[index] = original[index] + sign*nudge*along[index]
		}
		if losses[side], err = trainer.Loss(t.Context(), examples); err != nil {
			t.Fatal(err)
		}
	}
	copy(trainer.weights, original)
	clear(trainer.gradients)
	if measured := (losses[0] - losses[1]) / (2 * nudge); math.Abs(measured-predicted) > 2e-2*max(math.Abs(measured), math.Abs(predicted)) {
		t.Fatalf("the loss changes by %.6g a unit along a direction and the trainer's gradient says %.6g", measured, predicted)
	}
	before, err := trainer.Loss(t.Context(), examples)
	if err != nil {
		t.Fatal(err)
	}
	for range 40 {
		if _, err := trainer.Step(t.Context(), examples); err != nil {
			t.Fatal(err)
		}
	}
	after, err := trainer.Loss(t.Context(), examples)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss %.6g before, %.6g after 40 steps; last gradient %.4g", before, after, trainer.GradientL2())
	if !(after < before/2) || !(trainer.GradientL2() > 0) {
		t.Fatalf("40 steps took the loss from %.6g to %.6g", before, after)
	}
	for index, value := range decoder.predWeight {
		if value != frozen[index] {
			t.Fatal("training the bridge moved the decoder")
		}
	}
	// What it saves is the bridge it trained.
	path := filepath.Join(t.TempDir(), "bridge.safetensors")
	if err := trainer.SaveWeights(path); err != nil {
		t.Fatal(err)
	}
	file, err := safetensors.OpenFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	saved, err := safetensors.ReadF32(file.Tensors[bridgeWeightTensor])
	if err != nil {
		t.Fatal(err)
	}
	trained := trainer.Bridge()
	if len(saved) != len(trained.Weight) || len(trained.Bias) != decoder.latent {
		t.Fatalf("saved %d weights for a bridge of %d and %d offsets", len(saved), len(trained.Weight), len(trained.Bias))
	}
	for index, value := range saved {
		if value != trained.Weight[index] {
			t.Fatalf("saved weight %d is %v, the trained bridge holds %v", index, value, trained.Weight[index])
		}
	}
	if _, err := trainer.Loss(t.Context(), nil); err == nil {
		t.Fatal("a loss was given for no examples")
	}
	if _, err := NewBridgeTrainer(nil, config); err == nil {
		t.Fatal("a trainer was made for no pair")
	}
}
