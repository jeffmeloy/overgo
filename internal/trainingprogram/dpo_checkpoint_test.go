package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hostoptimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestDPOCheckpointAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, name)
	}
	policy := id(artifact.KindModel, "dpo-policy")
	reference := id(artifact.KindModel, "dpo-reference")
	dataset := id(artifact.KindDataset, "dpo-dataset")
	split := id(artifact.KindDatasetShard, "dpo-split")
	processor := id(artifact.KindProfile, "dpo-processor")
	muonPlan, err := hostoptimizer.CompilePlan(1, []hostoptimizer.GroupSpec{{Name: "weight", End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective: ObjectiveDPO,
		Operators: []OperatorSpec{
			{ID: ObjectiveOperatorForward, Phase: PhaseForward},
			{ID: ObjectiveOperatorBackward, Phase: PhaseBackward},
			{ID: ObjectiveOperatorMuon, Phase: PhaseOptimize},
		},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  muonPlan, Preference: &PreferencePolicy{Reference: reference, Scale: 0.1},
	})
	if err != nil {
		t.Fatal(err)
	}
	profile := func(name string) artifact.ID { return id(artifact.KindProfile, name) }
	runSpec := RunSpec{
		Recipe: id(artifact.KindRecipe, "dpo-recipe"), Initial: InitialStateSpec{Model: policy},
		Dataset: dataset, Split: split,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{processor}, Program: program,
		Policies: PolicySpec{
			Objective: profile("dpo-objective"), Precision: profile("dpo-precision"),
			Placement: profile("dpo-placement"), Memory: profile("dpo-memory"), Optimizer: BuiltinOptimizerPolicy().ID,
			Checkpoint: profile("dpo-checkpoint"), Evaluation: profile("dpo-evaluation"),
			Promotion: profile("dpo-promotion"),
		},
	}
	initial, err := CompileTrainingRunPlan(runSpec)
	if err != nil {
		t.Fatal(err)
	}
	config := hostoptimizer.Config{BaseLearningRate: 0.1, Momentum: 0.9, Steps: 1, Schedule: hostoptimizer.ScheduleConstant}
	opt, err := hostoptimizer.New([]float32{1}, []float32{0}, muonPlan, config)
	if err != nil {
		t.Fatal(err)
	}
	stream := id(artifact.KindProfile, "dpo-stream")
	algorithm := id(artifact.KindProfile, "counter-rng")
	checkpoint, err := NewCheckpoint(CheckpointSpec{
		RunPlan: initial.ID(), Program: program.ID(), Model: policy,
		Dataset: dataset, Split: split, Stream: DatasetState{Identity: stream, Position: 7},
		Optimizer: opt.Snapshot(), ParameterCount: 1,
		RNG: []RNGState{
			{Name: "augmentation", Algorithm: algorithm},
			{Name: "data", Algorithm: algorithm, Counter: 7},
		},
		Processors: []artifact.ID{processor},
		Lineage:    []LineageParent{{Artifact: reference, Relation: artifact.RelationDependsOn}},
	}, id(artifact.KindTensorSet, "dpo-weights"))
	if err != nil {
		t.Fatal(err)
	}
	runSpec.Initial = InitialStateSpec{Checkpoint: checkpoint.ID()}
	resumed, err := CompileTrainingRunPlan(runSpec)
	if err != nil {
		t.Fatal(err)
	}
	authority := ResumeAuthority{Model: policy, Stream: checkpoint.Stream, OptimizerPlan: muonPlan.Identity()}
	if err := ValidateResume(resumed, checkpoint, authority); err != nil {
		t.Fatal(err)
	}
	changedProgram, err := CompileTrainingProgram(ProgramSpec{
		Objective: program.Objective(), Operators: program.Operators(), Parameters: program.Parameters(),
		Optimizer: muonPlan, Preference: &PreferencePolicy{Reference: id(artifact.KindModel, "other-reference"), Scale: 0.1},
	})
	if err != nil {
		t.Fatal(err)
	}
	changed := runSpec
	changed.Program = changedProgram
	changedPlan, err := CompileTrainingRunPlan(changed)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateResume(changedPlan, checkpoint, authority); err == nil {
		t.Fatal("accepted checkpoint under changed DPO authority")
	}
}
