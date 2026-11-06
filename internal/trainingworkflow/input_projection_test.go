package trainingworkflow

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func TestInputProjectionResumeRefusals(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	profile := func(name string) artifact.ID { return id(artifact.KindProfile, name) }
	config := testcheck.Must(trainingprogram.BuiltinOptimizerPolicy().Config(4))(t)
	adapter := testcheck.Must(adaptertrain.NewLinearCTC(adaptertrain.LinearCTCSpec{OutputWeight: []float32{1, 0, 0, 1, -1, -1}, Width: 2, Vocabulary: 3, MaxFrames: 3, MaxTargets: 1, Blank: 0, MemoryBytes: 1 << 20, Optimizer: config}))(t)
	execution := testcheck.Must(adapter.Bind(t.Context()))(t)
	example := adaptertrain.LinearCTCExample{Hidden: []float32{1, 0, 0, 1, 1, 1}, Targets: []int{1}, Frames: 3}
	testcheck.NoError(t, execution.Run(&example))
	binding := adaptertrain.InputProjectionBinding{BaseRecipe: id(artifact.KindRecipe, "base"), TargetTransform: profile("transform")}
	runSpec := trainingprogram.RunSpec{Recipe: id(artifact.KindRecipe, "training-recipe"), Initial: trainingprogram.InitialStateSpec{Model: id(artifact.KindModel, "base")},
		Dataset: id(artifact.KindDataset, "corpus"), Split: id(artifact.KindDatasetShard, "split"), Processors: []artifact.ID{profile("processor"), binding.TargetTransform},
		Signature: recipecontract.ModalitySignature{Inputs: []recipecontract.Modality{recipecontract.ModalityAudio}, Outputs: []recipecontract.Modality{recipecontract.ModalityText}},
		Policies: trainingprogram.PolicySpec{Objective: profile("ctc"), Precision: profile("precision"), Placement: profile("placement"), Memory: profile("memory"),
			Optimizer: trainingprogram.BuiltinOptimizerPolicy().ID, Checkpoint: profile("checkpoint"), Evaluation: profile("evaluation"), Promotion: profile("promotion")}, Program: adapter.Program()}
	plan := testcheck.Must(trainingprogram.CompileTrainingRunPlan(runSpec))(t)
	spec := trainingprogram.CheckpointSpec{RunPlan: plan.ID(), Program: adapter.Program().ID(), Model: runSpec.Initial.Model, Dataset: runSpec.Dataset, Split: runSpec.Split,
		Processors: plan.Processors(), Stream: trainingprogram.DatasetState{Identity: profile("stream"), Position: 1},
		RNG: []trainingprogram.RNGState{{Name: "augmentation", Algorithm: profile("none")}, {Name: "data", Algorithm: profile("sampler"), Counter: 1}}}
	directory := filepath.Join(t.TempDir(), "checkpoint")
	checkpoint := testcheck.Must(PublishInputProjection(directory, adapter, binding, spec))(t)
	if _, err := PublishInputProjection(directory, adapter, binding, spec); err == nil {
		t.Fatal("existing checkpoint overwritten")
	}
	runSpec.Initial = trainingprogram.InitialStateSpec{Checkpoint: checkpoint.ID()}
	plan = testcheck.Must(trainingprogram.CompileTrainingRunPlan(runSpec))(t)
	authority := trainingprogram.ResumeAuthority{Model: spec.Model, Stream: spec.Stream, OptimizerPlan: adapter.Program().OptimizerIdentity()}
	beforeWeights := adapter.WeightSnapshot()
	beforeState := testcheck.Must(adapter.OptimizerSnapshot())(t)
	for _, name := range []string{"base", "transform", "model", "stream-identity", "stream-position", "optimizer", "rng-missing", "rng-counter", "weights-only"} {
		t.Run(name, func(t *testing.T) {
			candidateBinding, candidateAuthority, path := binding, authority, directory
			rng := slices.Clone(spec.RNG)
			switch name {
			case "base":
				candidateBinding.BaseRecipe = id(artifact.KindRecipe, "different")
			case "transform":
				candidateBinding.TargetTransform = profile("different")
			case "model":
				candidateAuthority.Model = id(artifact.KindModel, "different")
			case "stream-identity":
				candidateAuthority.Stream.Identity = profile("different")
			case "stream-position":
				candidateAuthority.Stream.Position++
			case "optimizer":
				candidateAuthority.OptimizerPlan = "different"
			case "rng-missing":
				rng = rng[:1]
			case "rng-counter":
				rng[1].Counter++
			case "weights-only":
				path = t.TempDir()
				testcheck.NoError(t, adapter.SaveWeights(path, binding))
			}
			if _, err := RestoreInputProjection(path, adapter, candidateBinding, plan, candidateAuthority, rng); err == nil {
				t.Fatal("invalid resume accepted")
			}
			afterState, err := adapter.OptimizerSnapshot()
			if err != nil || !reflect.DeepEqual(beforeState, afterState) || !slices.Equal(beforeWeights, adapter.WeightSnapshot()) {
				t.Fatalf("refusal changed live state: %v", err)
			}
		})
	}
	if _, err := RestoreInputProjection(directory, adapter, binding, plan, authority, spec.RNG); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, os.Remove(filepath.Join(directory, trainingprogram.CheckpointWeights)))
	if _, err := checkpoint.Batch("incomplete", directory); err == nil {
		t.Fatal("incomplete publication accepted")
	}
	if _, err := RestoreInputProjection(directory, adapter, binding, plan, authority, spec.RNG); err == nil {
		t.Fatal("missing weights restarted")
	}
}
