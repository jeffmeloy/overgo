package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestGUIResumeRequiresExactAuthority(t *testing.T) {
	id := func(kind artifact.Kind, value string) artifact.ID { return testutil.ArtifactID(t, kind, value) }
	optimizerPlan, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective: ObjectiveTokenPrediction,
		Operators: []OperatorSpec{
			{ID: "forward", Phase: PhaseForward}, {ID: "backward", Phase: PhaseBackward}, {ID: "optimize", Phase: PhaseOptimize},
		},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  optimizerPlan,
	})
	if err != nil {
		t.Fatal(err)
	}
	checkpointID := id(artifact.KindCheckpoint, "resume-checkpoint")
	datasetID := id(artifact.KindDataset, "resume-dataset")
	splitID := id(artifact.KindDatasetShard, "resume-split")
	streamID := id(artifact.KindDatasetShard, "resume-stream")
	processorID := id(artifact.KindProfile, "resume-processor")
	checkpoint := Checkpoint{
		id: checkpointID, Program: program.ID(), Dataset: datasetID, Split: splitID,
		Stream: DatasetState{Identity: streamID}, Processors: []artifact.ID{processorID},
	}
	profile := func(value string) artifact.ID { return id(artifact.KindProfile, value) }
	plan, err := CompileTrainingRunPlan(RunSpec{
		Recipe: id(artifact.KindRecipe, "resume-recipe"), Initial: InitialStateSpec{Checkpoint: checkpointID},
		Dataset: datasetID, Split: splitID,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{processorID}, Program: program,
		Policies: PolicySpec{
			Objective: profile("objective"), Precision: profile("precision"), Placement: profile("placement"),
			Memory: profile("memory"), Optimizer: BuiltinOptimizerPolicy().ID,
			Checkpoint: profile("checkpoint"), Evaluation: profile("evaluation"), Promotion: profile("promotion"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateResume(plan, checkpoint, streamID); err != nil {
		t.Fatal(err)
	}
	if err := ValidateResume(plan, checkpoint, id(artifact.KindDatasetShard, "other-stream")); err == nil {
		t.Fatal("different stream authority admitted")
	}
	checkpoint.Processors = []artifact.ID{profile("other-processor")}
	if err := ValidateResume(plan, checkpoint, streamID); err == nil {
		t.Fatal("different processor authority admitted")
	}
}
