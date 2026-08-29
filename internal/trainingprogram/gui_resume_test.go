package trainingprogram

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestQuantileRouterCheckpointResumeIsExactAndMismatchRefuses(t *testing.T) {
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
	modelID := id(artifact.KindModel, "resume-model")
	datasetID := id(artifact.KindDataset, "resume-dataset")
	splitID := id(artifact.KindDatasetShard, "resume-split")
	streamID := id(artifact.KindDatasetShard, "resume-stream")
	processorID := id(artifact.KindProfile, "resume-processor")
	projectorID := id(artifact.KindProjector, "resume-projector")
	codecID := id(artifact.KindProfile, "resume-codec")
	config, err := BuiltinOptimizerPolicy().Config(optimizerPlan.ParameterCount())
	if err != nil {
		t.Fatal(err)
	}
	state, err := optimizer.New([]float32{1}, []float32{0}, optimizerPlan, config)
	if err != nil {
		t.Fatal(err)
	}
	algorithmID := id(artifact.KindProfile, "resume-rng")
	routerPolicy := id(artifact.KindRecipe, "router-policy")
	routerState := RouterControllerState{Policy: routerPolicy, Stratum: splitID, Layer: 3, NextBias: []float32{-0.25, 0.25}}
	checkpoint, err := NewCheckpoint(CheckpointSpec{
		RunPlan: id(artifact.KindRecipe, "prior-run-plan"), Program: program.ID(), Model: modelID,
		Dataset: datasetID, Split: splitID, Stream: DatasetState{Identity: streamID},
		Optimizer: state.Snapshot(), ParameterCount: len(state.Snapshot().Momentum),
		RNG: []RNGState{
			{Name: "augmentation", Algorithm: algorithmID},
			{Name: "data", Algorithm: algorithmID},
		},
		Processors: []artifact.ID{processorID}, Projectors: []artifact.ID{projectorID}, Codecs: []artifact.ID{codecID},
		Lineage:           []LineageParent{{Artifact: modelID, Relation: artifact.RelationTrainedFrom}},
		RouterControllers: []RouterControllerState{routerState},
	}, id(artifact.KindTensorSet, "resume-weights"))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := checkpoint.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := ParseCheckpoint(encoded)
	if err != nil || restored.ID() != checkpoint.ID() || len(restored.RouterControllers) != 1 ||
		!slices.Equal(restored.RouterControllers[0].NextBias, routerState.NextBias) || len(restored.ArtifactLineage()) != 3 {
		t.Fatalf("router checkpoint round trip = (%+v, %v)", restored.RouterControllers, err)
	}
	duplicate := checkpoint
	duplicate.RouterControllers = append(cloneRouterControllerStates(checkpoint.RouterControllers), RouterControllerState{
		Policy: routerPolicy, Stratum: splitID, Layer: 3, NextBias: []float32{0},
	})
	if err := canonicalizeCheckpoint(&duplicate); err == nil {
		t.Fatal("duplicate router controller key admitted with different expert geometry")
	}
	nonfinite := checkpoint
	nonfinite.RouterControllers = cloneRouterControllerStates(checkpoint.RouterControllers)
	nonfinite.RouterControllers[0].NextBias[0] = float32(math.Inf(1))
	if err := canonicalizeCheckpoint(&nonfinite); err == nil {
		t.Fatal("non-finite router controller state admitted")
	}
	checkpointID := checkpoint.ID()
	profile := func(value string) artifact.ID { return id(artifact.KindProfile, value) }
	plan, err := CompileTrainingRunPlan(RunSpec{
		Recipe: id(artifact.KindRecipe, "resume-recipe"), Initial: InitialStateSpec{Checkpoint: checkpointID},
		Dataset: datasetID, Split: splitID,
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{processorID}, Projectors: []artifact.ID{projectorID}, Codecs: []artifact.ID{codecID}, Program: program,
		Policies: PolicySpec{
			Objective: profile("objective"), Precision: profile("precision"), Placement: profile("placement"),
			Memory: profile("memory"), Optimizer: BuiltinOptimizerPolicy().ID,
			Checkpoint: profile("checkpoint"), Evaluation: profile("evaluation"), Promotion: profile("promotion"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	authority := ResumeAuthority{
		Model: modelID, Stream: checkpoint.Stream, OptimizerPlan: optimizerPlan.Identity(),
		RouterControllers: []RouterControllerAuthority{{Policy: routerPolicy, Stratum: splitID, Layer: 3, Experts: 2}},
	}
	if err := ValidateResume(plan, checkpoint, authority); err != nil {
		t.Fatal(err)
	}
	otherStream := authority
	otherStream.Stream.Identity = id(artifact.KindDatasetShard, "other-stream")
	if err := ValidateResume(plan, checkpoint, otherStream); err == nil {
		t.Fatal("different stream authority admitted")
	}
	missingController := authority
	missingController.RouterControllers = nil
	if err := ValidateResume(plan, checkpoint, missingController); err == nil {
		t.Fatal("missing router controller authority admitted")
	}
	wrongExperts := authority
	wrongExperts.RouterControllers = slices.Clone(authority.RouterControllers)
	wrongExperts.RouterControllers[0].Experts++
	if err := ValidateResume(plan, checkpoint, wrongExperts); err == nil {
		t.Fatal("incompatible router expert inventory admitted")
	}
	mutations := map[string]func(*Checkpoint){
		"model":     func(value *Checkpoint) { value.Model = id(artifact.KindModel, "other-model") },
		"weights":   func(value *Checkpoint) { value.Weights = id(artifact.KindTensorSet, "other-weights") },
		"stream":    func(value *Checkpoint) { value.Stream.Position++ },
		"optimizer": func(value *Checkpoint) { value.Optimizer.PlanIdentity = "other-plan" },
		"rng":       func(value *Checkpoint) { value.RNG[0].Counter++ },
		"processor": func(value *Checkpoint) { value.Processors[0] = profile("other-processor") },
		"projector": func(value *Checkpoint) { value.Projectors[0] = id(artifact.KindProjector, "other-projector") },
		"codec":     func(value *Checkpoint) { value.Codecs[0] = profile("other-codec") },
		"router":    func(value *Checkpoint) { value.RouterControllers[0].NextBias[0] += 0.125 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			candidate := checkpoint
			candidate.Optimizer.Momentum = append([]float64(nil), checkpoint.Optimizer.Momentum...)
			candidate.RNG = append([]RNGState(nil), checkpoint.RNG...)
			candidate.Processors = append([]artifact.ID(nil), checkpoint.Processors...)
			candidate.Projectors = append([]artifact.ID(nil), checkpoint.Projectors...)
			candidate.Codecs = append([]artifact.ID(nil), checkpoint.Codecs...)
			candidate.Lineage = append([]LineageParent(nil), checkpoint.Lineage...)
			candidate.RouterControllers = cloneRouterControllerStates(checkpoint.RouterControllers)
			mutate(&candidate)
			if err := ValidateResume(plan, candidate, authority); err == nil {
				t.Fatal("changed checkpoint authority admitted")
			}
		})
	}
}
