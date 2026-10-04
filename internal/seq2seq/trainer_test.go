package seq2seq

import (
	"math"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
	"overgo/internal/trainingprogram"
)

// TestTrainerCheckpointRoundTripIsBitExact steps a tiny encoder-decoder
// trainer once, publishes its parameters and optimizer state as a
// checkpoint, and reads both back bit for bit under every parameter name.
func TestTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	model := tinyModel()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(model.TrainableParameterCount())
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(model, config)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if _, err := trainer.Step(TrainingPair{Source: []int{3, 4}, DecoderInput: []int{1, 5}, Targets: []int{5, 1}}); err != nil {
		t.Fatal(err)
	}
	if !(trainer.GradientL2() > 0) {
		t.Fatalf("step gradient norm %g, want a gradient", trainer.GradientL2())
	}
	trained := make(map[string][]float32, len(trainer.bindings))
	for _, binding := range trainer.bindings {
		trained[binding.name] = trainer.weights[binding.span.start:binding.span.end]
	}
	checkpointtest.RequireRoundTrip(t, trainer.Program(), trainer, trained)
}

// tinyModel is a deterministic encoder-decoder of one encoder and two
// decoder layers over an eight-token vocabulary.
func tinyModel() *Model {
	dims := Dims{
		Vocab: 8, DModel: 4, Heads: 2, KVHeads: 1, HeadDim: 2, EncoderLayers: 1, DecoderLayers: 2,
		RopeTheta: 10000, RMSEps: 1e-6, StartToken: 1, EOSToken: 1,
	}
	var drawn int
	matrix := func(count int) []uint16 {
		values := make([]uint16, count)
		for index := range values {
			drawn++
			values[index] = dtype.Float32ToBF16(float32(math.Sin(float64(drawn))) / 2)
		}
		return values
	}
	ones := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = 1
		}
		return values
	}
	qWidth, kvWidth := dims.Heads*dims.HeadDim, dims.KVHeads*dims.HeadDim
	block := func() attnBlock {
		return attnBlock{
			inNorm: ones(dims.DModel), q: matrix(qWidth * dims.DModel), k: matrix(kvWidth * dims.DModel),
			v: matrix(kvWidth * dims.DModel), o: matrix(dims.DModel * qWidth),
			qNorm: ones(dims.HeadDim), kNorm: ones(dims.HeadDim), gate: sigmoid(0),
		}
	}
	model := &Model{
		Dims: dims, embed: matrix(dims.Vocab * dims.DModel),
		encFinalNorm: ones(dims.DModel), decFinalNorm: ones(dims.DModel),
		invFreq:    hostmath.RopeInvFreq(dims.RopeTheta, dims.HeadDim),
		embedScale: float32(math.Sqrt(float64(dims.DModel))), scoreScale: float32(1 / math.Sqrt(float64(dims.HeadDim))),
	}
	for range dims.EncoderLayers {
		model.encoder = append(model.encoder, block())
	}
	for range dims.DecoderLayers {
		model.decoderSelf = append(model.decoderSelf, block())
		model.decoderCross = append(model.decoderCross, block())
	}
	return model
}
