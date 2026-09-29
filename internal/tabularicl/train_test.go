package tabularicl

import (
	"math"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
	"overgo/internal/testskip"
	"overgo/internal/trainingprogram"
)

func TestDecoderTrainerRealArtifacts(t *testing.T) {
	testskip.Short(t, "loads both TabFM heads")
	golden := readGolden(t)
	for _, task := range Tasks() {
		t.Run(task, func(t *testing.T) {
			var request Request
			for _, testCase := range golden.Cases {
				if testCase.Task == task {
					request = Request{
						Task: task, X: testCase.X, Y: testCase.Y, Rows: testCase.Rows,
						Cols: testCase.Cols, TrainRows: testCase.TrainRows,
					}
					break
				}
			}
			if request.Rows == 0 {
				t.Fatalf("%s fixture absent", task)
			}
			head, err := LoadHead(headDir(t, task))
			if err != nil {
				t.Fatal(err)
			}
			before := append([]float32(nil), head.decL1W...)
			trainer, err := NewTrainer(head, optimizer.Config{
				BaseLearningRate: 1e-5, Momentum: 0.9, Schedule: optimizer.ScheduleConstant,
			})
			if err != nil {
				t.Fatal(err)
			}
			result, stepErr := trainer.Step(request)
			closeErr := trainer.Close()
			if stepErr != nil || closeErr != nil {
				t.Fatalf("step=%v close=%v", stepErr, closeErr)
			}
			if trainer.program.Objective() != trainingprogram.ObjectiveTablePrediction ||
				trainer.ParameterCount() != len(head.decL0W)+len(head.decL0B)+len(head.decL1W)+len(head.decL1B) ||
				result.Step != 1 || result.LearningRate <= 0 || result.GradientL2 <= 0 ||
				math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) || slices.Equal(before, head.decL1W) {
				t.Fatalf("result=%+v parameters=%d unchanged=%t", result, trainer.ParameterCount(), slices.Equal(before, head.decL1W))
			}
			t.Logf("%s decoder update: loss=%.6f grad_l2=%.6g parameters=%d", task, result.Loss, result.GradientL2, trainer.ParameterCount())
			head, trainer = nil, nil
			runtime.GC()
		})
	}
}

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
	head := tinyRegressionHead()
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(head.TrainableParameterCount())
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := NewTrainer(head, config)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	if _, err := trainer.Step(Request{
		Task: TaskRegression, Rows: 4, Cols: 2, TrainRows: 2,
		X: []float32{0.1, 0.7, 0.4, 0.2, 0.9, 0.3, 0.6, 0.8}, Y: []float32{0.5, 1.5, 1, 2},
	}); err != nil {
		t.Fatal(err)
	}
	state := trainer.Snapshot()
	loaded, catalog := checkpointRoundTrip(t, trainer.Program(), state, trainer.SaveWeights)
	if state.Step != 1 || state.PlanIdentity != trainer.Program().OptimizerIdentity() ||
		!slices.ContainsFunc(state.Momentum, func(value float64) bool { return value != 0 }) {
		t.Fatalf("snapshot after one step = step %d plan %s, want step 1 of %s with momentum", state.Step, state.PlanIdentity, trainer.Program().OptimizerIdentity())
	}
	if loaded.Step != state.Step || loaded.Config != state.Config || loaded.PlanIdentity != state.PlanIdentity ||
		!slices.EqualFunc(loaded.Momentum, state.Momentum, func(left, right float64) bool { return math.Float64bits(left) == math.Float64bits(right) }) {
		t.Fatal("the checkpoint's optimizer state differs from the trainer's")
	}
	trained := map[string][]float32{
		decoderInputWeight: head.decL0W, decoderInputBias: head.decL0B,
		decoderOutputWeight: head.decL1W, decoderOutputBias: head.decL1B,
	}
	parameters := trainer.Program().Parameters()
	if len(catalog.Values) != len(trained) || len(parameters) != len(trained) {
		t.Fatalf("checkpoint holds %d tensors and the program %d, want the %d decoder tensors", len(catalog.Values), len(parameters), len(trained))
	}
	for _, parameter := range parameters {
		if !slices.Equal(catalog.Shapes[parameter.Name], []int{parameter.Rows, parameter.Cols}) ||
			!slices.EqualFunc(catalog.Values[parameter.Name], trained[parameter.Name], func(left, right float32) bool { return math.Float32bits(left) == math.Float32bits(right) }) {
			t.Fatalf("checkpoint tensor %q differs from the trained decoder", parameter.Name)
		}
	}
}

// checkpointRoundTrip publishes program's state and the weights write
// stages as a checkpoint under stand-in authority, and reads the state and
// weights back.
func checkpointRoundTrip(t *testing.T, program trainingprogram.TrainingProgram, state optimizer.State, write func(string) error) (optimizer.State, safetensors.F32Catalog) {
	t.Helper()
	id := func(kind artifact.Kind, name string) artifact.ID {
		value, err := artifact.IdentifyBytes(kind, []byte(name))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	model, rng := id(artifact.KindModel, "model"), id(artifact.KindProfile, "rng")
	target := filepath.Join(t.TempDir(), "checkpoint")
	if _, err := trainingprogram.PublishCheckpoint(target, trainingprogram.CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "run plan"), Program: program.ID(), Model: model,
		Dataset: id(artifact.KindDataset, "dataset"), Split: id(artifact.KindDatasetShard, "split"),
		Stream:    trainingprogram.DatasetState{Identity: id(artifact.KindDataset, "stream"), Position: uint64(state.Step)},
		Optimizer: state, ParameterCount: len(state.Momentum),
		RNG:        []trainingprogram.RNGState{{Name: "augmentation", Algorithm: rng}, {Name: "data", Algorithm: rng}},
		Processors: []artifact.ID{id(artifact.KindProfile, "processor")},
		Lineage:    []trainingprogram.LineageParent{{Artifact: model, Relation: artifact.RelationTrainedFrom}},
	}, func(stage string) error { return write(filepath.Join(stage, trainingprogram.CheckpointWeights)) }); err != nil {
		t.Fatal(err)
	}
	loaded, err := trainingprogram.LoadCheckpoint(target)
	if err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenFile(filepath.Join(target, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	catalog, err := source.MaterializeF32(safetensors.F32Selection{RetainShapes: true})
	if err != nil {
		t.Fatal(err)
	}
	return loaded.Optimizer, catalog
}
