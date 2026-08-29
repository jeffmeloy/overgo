package trainingprogram

import (
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
)

func TestObjectiveMatrixUsesOnlyStoredCorpusBoundObjectives(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	text := objectiveFixture(t, recipecontract.ModalityText, recipecontract.ModalityText, false)
	image := objectiveFixture(t, recipecontract.ModalityImage, recipecontract.ModalityText, true)
	descriptors := objectiveReferences(text)
	descriptors = append(descriptors, objectiveReferences(image)...)
	contents := make([]artifact.Content, 0, 2)
	for _, objective := range []ObjectiveDocument{text, image} {
		content, err := objective.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "objectives", Artifacts: descriptors, Contents: contents}); err != nil {
		t.Fatal(err)
	}
	rows, err := CompileObjectiveMatrix(ctx, store, []artifact.ID{image.ID, text.ID})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 36 {
		t.Fatalf("objective rows=%d, want 36", len(rows))
	}
	trainable := map[string]artifact.ID{"text->text": text.ID, "image->text": image.ID}
	for _, row := range rows {
		key := objectivePairKey(row.Signature)
		want, ok := trainable[key]
		if ok {
			if row.Disposition != ObjectiveTrainable || row.Objective != want || row.Reason != "" {
				t.Fatalf("trainable row %s differs: %+v", key, row)
			}
			delete(trainable, key)
			continue
		}
		if row.Disposition != ObjectiveRefused || row.Objective.Valid() || row.Reason == "" {
			t.Fatalf("refusal row %s differs: %+v", key, row)
		}
	}
	if len(trainable) != 0 {
		t.Fatalf("trainable rows absent: %v", trainable)
	}
}

func TestRepositoryObjectiveBindsTrainingRun(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objective := objectiveFixture(t, recipecontract.ModalityImage, recipecontract.ModalityText, true)
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "run-objective", Artifacts: objectiveReferences(objective), Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	muon, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective:  ObjectiveTokenPrediction,
		Operators:  []OperatorSpec{{ID: "forward", Phase: PhaseForward}, {ID: "backward", Phase: PhaseBackward}, {ID: "muon", Phase: PhaseOptimize}},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}}, Optimizer: muon,
	})
	if err != nil {
		t.Fatal(err)
	}
	identity := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	profile := func(name string) artifact.ID { return identity(artifact.KindProfile, name) }
	modelID := identity(artifact.KindModel, "model")
	spec := RunSpec{
		Initial: InitialStateSpec{Model: modelID},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature,
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Policies: PolicySpec{
			Objective: objective.ID, Precision: profile("precision"), Placement: profile("placement"),
			Memory: profile("memory"), Optimizer: BuiltinOptimizerPolicy().ID,
			Checkpoint: profile("checkpoint"), Evaluation: objective.Evaluation,
			Promotion: profile("promotion"),
		},
		Program: program,
	}
	definition := trainingRunRecipeFixture(t, recipe.TaskTraining, modelID, spec.Policies)
	spec.Recipe = definition.ID
	definitionContent, err := definition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "run-policies", Artifacts: []artifact.Descriptor{
		{ID: spec.Policies.Precision}, {ID: spec.Policies.Placement}, {ID: spec.Policies.Memory},
		{ID: spec.Policies.Checkpoint}, {ID: spec.Policies.Promotion},
	}, Contents: []artifact.Content{mustOptimizerPolicyContent(t), definitionContent}}); err != nil {
		t.Fatal(err)
	}
	if _, err := CompileTrainingRunPlanFromRepository(ctx, store, spec); err != nil {
		t.Fatal(err)
	}
	spec.Signature.Inputs[0] = recipecontract.ModalityAudio
	if _, err := CompileTrainingRunPlanFromRepository(ctx, store, spec); err == nil {
		t.Fatal("run with a different modality signature accepted")
	}
}

func TestReferenceAdmissionRequiresSemanticRelevance(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	objective := objectiveFixture(t, recipecontract.ModalityText, recipecontract.ModalityText, false)
	objectiveContent, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	muon, err := optimizer.CompilePlan(1, []optimizer.GroupSpec{{Name: "weight", Start: 0, End: 1, Rows: 1, Cols: 1}})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileTrainingProgram(ProgramSpec{
		Objective: ObjectiveTokenPrediction,
		Operators: []OperatorSpec{
			{ID: "forward", Phase: PhaseForward},
			{ID: "backward", Phase: PhaseBackward},
			{ID: "muon", Phase: PhaseOptimize},
		},
		Parameters: []ParameterSpec{{Name: "weight", Rows: 1, Cols: 1, Trainable: true}},
		Optimizer:  muon,
	})
	if err != nil {
		t.Fatal(err)
	}
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	profile := func(name string) artifact.ID { return id(artifact.KindProfile, name) }
	modelID := id(artifact.KindModel, "bound-model")
	policies := PolicySpec{
		Objective: objective.ID, Precision: profile("bound-precision"), Placement: profile("bound-placement"),
		Memory: profile("bound-memory"), Optimizer: BuiltinOptimizerPolicy().ID,
		Checkpoint: profile("bound-checkpoint"), Evaluation: objective.Evaluation,
		Promotion: profile("bound-promotion"),
	}
	exact := trainingRunRecipeFixture(t, recipe.TaskTraining, modelID, policies)
	foreignPolicies := policies
	foreignPolicies.Precision = profile("foreign-precision")
	foreignDataset := id(artifact.KindDataset, "foreign-dataset")
	foreignSplit := id(artifact.KindDatasetShard, "foreign-split")
	foreignPolicyRecipe := trainingRunRecipeFixture(t, recipe.TaskTraining, modelID, foreignPolicies)
	foreignModelRecipe := trainingRunRecipeFixture(t, recipe.TaskTraining, id(artifact.KindModel, "foreign-model"), policies)
	foreignTaskRecipe := trainingRunRecipeFixture(t, recipe.TaskInference, modelID, policies)

	contents := []artifact.Content{objectiveContent, mustOptimizerPolicyContent(t)}
	for _, definition := range []recipe.Definition{exact, foreignPolicyRecipe, foreignModelRecipe, foreignTaskRecipe} {
		content, contentErr := definition.ArtifactContent()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		contents = append(contents, content)
	}
	descriptors := objectiveReferences(objective)
	for _, reference := range []artifact.ID{
		policies.Precision, policies.Placement, policies.Memory, policies.Checkpoint, policies.Promotion,
		foreignPolicies.Precision, foreignDataset, foreignSplit,
	} {
		descriptors = append(descriptors, artifact.Descriptor{ID: reference})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "semantic-reference-fixture", Artifacts: descriptors, Contents: contents}); err != nil {
		t.Fatal(err)
	}

	spec := RunSpec{
		Recipe: exact.ID, Initial: InitialStateSpec{Model: modelID},
		Dataset: objective.Dataset, Split: objective.Split, Signature: objective.Signature,
		Processors: objective.Processors, Projectors: objective.Projectors, Codecs: objective.Codecs,
		Policies: policies, Program: program,
	}
	if _, err := CompileTrainingRunPlanFromRepository(ctx, store, spec); err != nil {
		t.Fatalf("exact recipe authority refused: %v", err)
	}

	for name, mutate := range map[string]func(*RunSpec){
		"foreign recipe policy": func(value *RunSpec) { value.Recipe = foreignPolicyRecipe.ID },
		"foreign run policy":    func(value *RunSpec) { value.Policies.Precision = foreignPolicies.Precision },
		"foreign dataset":       func(value *RunSpec) { value.Dataset = foreignDataset },
		"foreign split":         func(value *RunSpec) { value.Split = foreignSplit },
		"foreign model":         func(value *RunSpec) { value.Recipe = foreignModelRecipe.ID },
		"foreign task":          func(value *RunSpec) { value.Recipe = foreignTaskRecipe.ID },
	} {
		t.Run(name, func(t *testing.T) {
			foreign := spec
			mutate(&foreign)
			if _, err := CompileTrainingRunPlanFromRepository(ctx, store, foreign); err == nil {
				t.Fatal("well-formed foreign authority entered the training run")
			}
		})
	}
}

func trainingRunRecipeFixture(t *testing.T, task recipe.Task, model artifact.ID, policies PolicySpec) recipe.Definition {
	t.Helper()
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: model},
		{Role: recipe.DependencyObjective, Artifact: policies.Objective},
		{Role: recipe.DependencyPrecision, Artifact: policies.Precision},
		{Role: recipe.DependencyPlacement, Artifact: policies.Placement},
		{Role: recipe.DependencyMemory, Artifact: policies.Memory},
		{Role: recipe.DependencyOptimizer, Artifact: policies.Optimizer},
		{Role: recipe.DependencyCheckpointPolicy, Artifact: policies.Checkpoint},
		{Role: recipe.DependencyEvaluation, Artifact: policies.Evaluation},
		{Role: recipe.DependencyPromotion, Artifact: policies.Promotion},
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		task, dependencies,
		[]recipe.Node{{ID: "train", Module: "train", Placement: recipe.PlacementHost}}, nil, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "train", Port: "checkpoint"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	return definition
}

func mustOptimizerPolicyContent(t *testing.T) artifact.Content {
	t.Helper()
	content, err := BuiltinOptimizerPolicy().Content()
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func TestCompileObjectiveProgramDerivesOptimizerParameters(t *testing.T) {
	plan, err := optimizer.CompilePlan(6, []optimizer.GroupSpec{
		{Name: "matrix", Start: 0, End: 4, Rows: 2, Cols: 2},
		{Name: "frozen", Start: 4, End: 6, Rows: 1, Cols: 2, Frozen: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileObjectiveProgram(ObjectiveTokenPrediction, nil, plan)
	if err != nil {
		t.Fatal(err)
	}
	parameters := program.Parameters()
	if len(parameters) != 2 || parameters[0].Name != "matrix" ||
		parameters[0].Rows != 2 || parameters[0].Cols != 2 || !parameters[0].Trainable ||
		parameters[1].Name != "frozen" || parameters[1].Trainable {
		t.Fatalf("derived parameters = %+v", parameters)
	}
}

func TestObjectiveMatrixRefusesUnstoredEvidence(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	objective := objectiveFixture(t, recipecontract.ModalityText, recipecontract.ModalityAudio, false)
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "objective-only", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	if _, err := CompileObjectiveMatrix(ctx, store, []artifact.ID{objective.ID}); err == nil {
		t.Fatal("objective with unstored corpus/evidence references accepted")
	}
}

func objectiveFixture(t *testing.T, input, output recipecontract.Modality, projector bool) ObjectiveDocument {
	t.Helper()
	ids := func(kind artifact.Kind, label string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte(label+"/"+string(input)+"/"+string(output)))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	var projectors []artifact.ID
	if projector {
		projectors = []artifact.ID{ids(artifact.KindProjector, "projector")}
	}
	objective, err := NewObjective(ObjectiveSpec{
		Name: string(input) + "-to-" + string(output), Kind: fixtureObjectiveKind(input, output),
		Signature: recipecontract.ModalitySignature{
			Inputs: []recipecontract.Modality{input}, Outputs: []recipecontract.Modality{output},
		},
		Dataset: ids(artifact.KindDataset, "dataset"), Split: ids(artifact.KindDatasetShard, "split"),
		Processors: []artifact.ID{ids(artifact.KindProfile, "processor")}, Projectors: projectors,
		Loss: ids(artifact.KindProfile, "loss"), Evaluation: ids(artifact.KindProfile, "evaluation"),
		Metric:   fixtureMetric(output),
		Evidence: []artifact.ID{ids(artifact.KindEvidence, "evidence")}, Authority: ObjectiveAdaptive,
	})
	if err != nil {
		t.Fatal(err)
	}
	return objective
}

func fixtureObjectiveKind(input, output recipecontract.Modality) ObjectiveKind {
	if output == recipecontract.ModalityText {
		return ObjectiveTokenPrediction
	}
	if input == recipecontract.ModalityText && output == recipecontract.ModalityAudio {
		return ObjectiveLatentL2
	}
	return ObjectiveFlowMatching
}

func objectiveReferences(objective ObjectiveDocument) []artifact.Descriptor {
	ids := []artifact.ID{objective.Dataset, objective.Split, objective.Loss, objective.Evaluation}
	ids = append(ids, objective.Processors...)
	ids = append(ids, objective.Projectors...)
	ids = append(ids, objective.Codecs...)
	ids = append(ids, objective.Evidence...)
	result := make([]artifact.Descriptor, len(ids))
	for index, id := range ids {
		result[index] = artifact.Descriptor{ID: id}
	}
	return result
}

func fixtureMetric(modality recipecontract.Modality) EvaluationMetric {
	switch modality {
	case recipecontract.ModalityText:
		return MetricTokenAccuracy
	case recipecontract.ModalityImage:
		return MetricImagePSNRSigned
	case recipecontract.ModalityAudio:
		return MetricAudioSNR
	case recipecontract.ModalityVideo:
		return MetricVideoPSNRSigned
	case recipecontract.ModalityTimeSeries:
		return MetricForecastMAE
	case recipecontract.ModalityTable:
		return MetricTableAccuracy
	default:
		return ""
	}
}

// TestObjectiveAuthorityLadder pins the graduated validation levels:
// declared < adaptive-evidence < approved, an AtLeast floor check, and
// rejection of an undeclared level.
func TestObjectiveAuthorityLadder(t *testing.T) {
	ascending := []ObjectiveAuthority{ObjectiveDeclared, ObjectiveAdaptive, ObjectiveApproved}
	for lower := range ascending {
		for higher := range ascending {
			meets := ascending[higher].AtLeast(ascending[lower])
			if want := higher >= lower; meets != want {
				t.Fatalf("%s.AtLeast(%s) = %t, want %t", ascending[higher], ascending[lower], meets, want)
			}
		}
	}
	if ObjectiveDeclared.AtLeast(ObjectiveApproved) {
		t.Fatal("declared must not satisfy an approved floor")
	}
	if !ValidObjectiveAuthority(ObjectiveDeclared) || ValidObjectiveAuthority(ObjectiveAuthority("mystery")) {
		t.Fatal("validity check wrong")
	}
	if ObjectiveAuthority("mystery").AtLeast(ObjectiveDeclared) {
		t.Fatal("an undeclared level must not meet any floor")
	}
}
