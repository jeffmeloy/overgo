package trainingprogram

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestCompiledTrainingAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, name)
	}
	rngAlgorithm := id(artifact.KindProfile, "counter-rng-v1")
	scratchSpec := ScratchSpec{
		Recipe:             id(artifact.KindRecipe, "scratch-recipe"),
		Dataset:            id(artifact.KindDataset, "workflow-corpus"),
		Split:              id(artifact.KindDatasetShard, "workflow-split"),
		DerivationProfile:  id(artifact.KindProfile, "adaptive-derivation-v1"),
		TopologyProfile:    id(artifact.KindProfile, "adaptive-causal-v1"),
		Tokenizer:          id(artifact.KindTokenizer, "rune-tokenizer"),
		ParameterManifest:  id(artifact.KindTensorInventory, "parameter-manifest"),
		InitializerProfile: id(artifact.KindProfile, "adaptive-uniform-v1"),
		InitializedModel:   id(artifact.KindModel, "initialized-model"),
		RNGStreams: []RNGStreamSpec{
			{Name: "split", Algorithm: rngAlgorithm, Seed: 11},
			{Name: "init", Algorithm: rngAlgorithm, Seed: 13},
			{Name: "data", Algorithm: rngAlgorithm, Seed: 17},
			{Name: "augmentation", Algorithm: rngAlgorithm, Seed: 19},
		},
	}
	scratch, err := CompileScratchConstruction(scratchSpec)
	if err != nil {
		t.Fatal(err)
	}
	if scratch.ID().Kind() != artifact.KindRecipe || scratch.InitializedModel() != scratchSpec.InitializedModel {
		t.Fatalf("scratch construction identity/model = %s / %s", scratch.ID(), scratch.InitializedModel())
	}

	muon, err := optimizer.CompilePlan(6, []optimizer.GroupSpec{
		{Name: "weight", Start: 0, End: 4, Rows: 2, Cols: 2},
		{Name: "bias", Start: 4, End: 6, Rows: 2, Cols: 1, Frozen: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	programSpec := ProgramSpec{
		Objective: ObjectiveTokenPrediction,
		Operators: []OperatorSpec{
			{ID: "batch", Phase: PhaseBatch},
			{ID: "forward", Phase: PhaseForward},
			{ID: "loss", Phase: PhaseLoss},
			{ID: "backward", Phase: PhaseBackward},
			{ID: "optimize", Phase: PhaseOptimize},
		},
		Parameters: []ParameterSpec{
			{Name: "weight", Rows: 2, Cols: 2, Trainable: true},
			{Name: "bias", Rows: 2, Cols: 1, Trainable: false},
		},
		Optimizer: muon,
	}
	program, err := CompileTrainingProgram(programSpec)
	if err != nil {
		t.Fatal(err)
	}
	if program.ID().Kind() != artifact.KindRecipe || program.Objective() != ObjectiveTokenPrediction || program.OptimizerIdentity() != muon.Identity() {
		t.Fatalf("program identity/objective/Muon = %s / %s / %s", program.ID(), program.Objective(), program.OptimizerIdentity())
	}

	profile := func(name string) artifact.ID { return id(artifact.KindProfile, name) }
	processorA, processorB := profile("processor-a"), profile("processor-b")
	runSpec := RunSpec{
		Recipe:  id(artifact.KindRecipe, "training-recipe"),
		Initial: InitialStateSpec{Scratch: &scratch},
		Dataset: scratchSpec.Dataset,
		Split:   scratchSpec.Split,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Processors: []artifact.ID{processorB, processorA},
		Policies: PolicySpec{
			Objective: profile("causal-objective"), Precision: profile("fp32"),
			Placement: profile("resident-device"), Memory: profile("resident-memory"), Optimizer: BuiltinOptimizerPolicy().ID,
			Checkpoint: profile("exact-checkpoint"), Evaluation: profile("heldout-evaluation"),
			Promotion: profile("champion-challenger"),
		},
		Program: program,
	}
	plan, err := CompileTrainingRunPlan(runSpec)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ID().Kind() != artifact.KindRecipe || plan.InitialMode() != InitialScratch ||
		plan.Model() != scratchSpec.InitializedModel || plan.Dataset() != scratchSpec.Dataset ||
		plan.Program().ID() != program.ID() {
		t.Fatalf("compiled plan differs: id=%s mode=%s model=%s dataset=%s program=%s", plan.ID(), plan.InitialMode(), plan.Model(), plan.Dataset(), plan.Program().ID())
	}
	if got := plan.Processors(); len(got) != 2 || got[0] != processorA || got[1] != processorB {
		t.Fatalf("processors not canonical: %v", got)
	}
	returned := plan.Processors()
	returned[0] = artifact.ID{}
	if plan.Processors()[0] != processorA {
		t.Fatal("plan exposed mutable processor authority")
	}
	signature := plan.Signature()
	signature.Inputs[0] = recipecontract.ModalityImage
	if plan.Signature().Inputs[0] != recipecontract.ModalityText {
		t.Fatal("plan exposed mutable modality authority")
	}

	t.Run("initial-state-exclusive", func(t *testing.T) {
		invalid := runSpec
		invalid.Initial.Model = id(artifact.KindModel, "pretrained")
		if _, err := CompileTrainingRunPlan(invalid); err == nil {
			t.Fatal("scratch plus pretrained initial state accepted")
		}
	})
	t.Run("rng-streams-independent", func(t *testing.T) {
		invalid := scratchSpec
		invalid.RNGStreams = append([]RNGStreamSpec(nil), scratchSpec.RNGStreams[:3]...)
		if _, err := CompileScratchConstruction(invalid); err == nil {
			t.Fatal("missing independent RNG stream accepted")
		}
	})
	t.Run("muon-manifest-bound", func(t *testing.T) {
		invalid := programSpec
		invalid.Parameters = append([]ParameterSpec(nil), programSpec.Parameters...)
		invalid.Parameters[0].Rows = 1
		if _, err := CompileTrainingProgram(invalid); err == nil {
			t.Fatal("parameter manifest differing from Muon plan accepted")
		}
	})
	t.Run("semantic-order-bound", func(t *testing.T) {
		invalid := programSpec
		invalid.Operators = append([]OperatorSpec(nil), programSpec.Operators...)
		invalid.Operators[1], invalid.Operators[3] = invalid.Operators[3], invalid.Operators[1]
		if _, err := CompileTrainingProgram(invalid); err == nil {
			t.Fatal("backward-before-forward program accepted")
		}
	})
	t.Run("objective-bound", func(t *testing.T) {
		invalid := programSpec
		invalid.Objective = ""
		if _, err := CompileTrainingProgram(invalid); err == nil {
			t.Fatal("program without objective accepted")
		}
	})
}

func TestBoundTrainingProgramOwnsOrder(t *testing.T) {
	plan, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "scalar", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective: ObjectiveTokenPrediction,
		Operators: []OperatorSpec{
			{ID: "forward", Phase: PhaseForward},
			{ID: "backward", Phase: PhaseBackward},
			{ID: "optimize", Phase: PhaseOptimize},
			{ID: "evaluate", Phase: PhaseEvaluate},
		},
		Parameters: []ParameterSpec{{Name: "scalar", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  plan,
	})
	if err != nil {
		t.Fatal(err)
	}
	type state struct{ order []string }
	binding := func(id string) Binding[state] {
		return Binding[state]{Operator: id, Execute: func(value *state) error {
			value.order = append(value.order, id)
			return nil
		}}
	}
	execution, err := Bind(program, []Binding[state]{binding("optimize"), binding("evaluate"), binding("backward"), binding("forward")})
	if err != nil {
		t.Fatal(err)
	}
	execution, err = execution.Select(PhaseForward, PhaseBackward, PhaseOptimize)
	if err != nil {
		t.Fatal(err)
	}
	value := state{}
	if err := execution.Run(&value); err != nil {
		t.Fatal(err)
	}
	want := []string{"forward", "backward", "optimize"}
	if !slices.Equal(value.order, want) {
		t.Fatalf("bound order = %v, want %v", value.order, want)
	}
	if _, err := Bind(program, []Binding[state]{binding("forward")}); err == nil {
		t.Fatal("partial binding accepted")
	}
	if _, err := Bind(program, []Binding[state]{binding("forward"), binding("backward"), binding("optimize"), binding("evaluate"), binding("foreign")}); err == nil {
		t.Fatal("foreign binding accepted")
	}
}
