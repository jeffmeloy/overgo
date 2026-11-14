package tabularicl

import (
	"testing"

	"overgo/internal/checkpointtest"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

// tinyRegressionHead builds a synthetic regression head whose towers carry
// no attention blocks, so the frozen feature program is its embeddings,
// projections and norms alone and the decoder trains over it.
func tinyRegressionHead() *Head {
	const embed, hidden = 2, 3
	seed := uint32(2463534242)
	fill := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			seed ^= seed << 13
			seed ^= seed >> 17
			seed ^= seed << 5
			values[index] = float32(seed%2000)/1000 - 1
		}
		return values
	}
	return &Head{
		Dims: Dims{EmbedDim: embed, NumFreq: 1, FeatureGroupSize: 1, NumCLS: 1, OutDim: 1},
		ff:   fill(1), wNum: fill(2 * embed), bNum: fill(embed),
		cellYL0W: fill(hidden), cellYL0B: fill(hidden), cellYL1W: fill(embed * hidden), cellYL1B: fill(embed),
		iclYL0W: fill(hidden), iclYL0B: fill(hidden), iclYL1W: fill(embed * hidden), iclYL1B: fill(embed),
		col1OutW: fill(embed * embed), col1OutB: fill(embed), col1LN: fill(embed),
		col2OutW: fill(embed * embed), col2OutB: fill(embed), col2LN: fill(embed),
		clsTokens: fill(embed),
		row1:      &encoder{outLN: fill(embed)}, row2: &encoder{outLN: fill(embed)}, icl: &encoder{outLN: fill(embed)},
		decL0W: fill(hidden * embed), decL0B: fill(hidden), decL1W: fill(hidden), decL1B: fill(1),
	}
}

// TestDecoderTrainerCheckpointRoundTripIsBitExact trains one decoder step,
// publishes the trainer's weights and optimizer state as a checkpoint, and
// reads both back bit for bit.
func TestDecoderTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	cudatest.Require(t)
	head := tinyRegressionHead()
	config := testcheck.Must(trainingprogram.BuiltinOptimizerPolicy().Config(head.TrainableParameterCount()))(t)
	trainer := testcheck.Must(NewTrainer(head, config))(t)
	defer trainer.Close()
	if _, err := trainer.Step(Request{
		Task: TaskRegression, Rows: 4, Cols: 2, TrainRows: 2,
		X: []float32{0.1, 0.7, 0.4, 0.2, 0.9, 0.3, 0.6, 0.8}, Y: []float32{0.5, 1.5, 1, 2},
	}); err != nil {
		t.Fatal(err)
	}
	checkpointtest.RequireRoundTrip(t, trainer.Program(), trainer, map[string][]float32{
		decoderInputWeight: head.decL0W, decoderInputBias: head.decL0B,
		decoderOutputWeight: head.decL1W, decoderOutputBias: head.decL1B,
	})
}
