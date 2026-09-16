package trainingworkflow

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

func TestTrainingWorkflowRequiresStoredAuthority(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 6,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("ab"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "missing active recipe")
	_, err = Execute(t.Context(), Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "output"), Steps: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "no active training recipe") {
		t.Fatalf("absent authority error = %v", err)
	}
}

func TestTrainingWorkflowExactResumeMatrix(t *testing.T) {
	t.Run("token", testTokenResume)
	t.Run("dpo", testDPOResume)
	t.Run("grpo", testGRPOResume)
}

func TestMoETrainingPublishesRouterObservationsWithoutChangingNumerics(t *testing.T) {
	executableCommit := strings.Repeat("ab", 20)
	stubExecutableCodeCommit(t, executableCommit)
	root, model, dataset, store, recipeID := moeWorkflowFixture(t)
	base := Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		Steps: 1, Host: true,
	}
	base.OutputDirectory = filepath.Join(root, "without-observation")
	want, err := Execute(t.Context(), base)
	if err != nil {
		t.Fatal(err)
	}
	base.OutputDirectory = filepath.Join(root, "with-observation")
	base.Observations = store
	got, err := Execute(t.Context(), base)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Losses, want.Losses) || got.Checkpoint.ID() != want.Checkpoint.ID() {
		t.Fatalf("instrumented numerics differ: losses=%v/%v checkpoints=%s/%s", got.Losses, want.Losses, got.Checkpoint.ID(), want.Checkpoint.ID())
	}
	wantModel, err := densecausal.Load(filepath.Join(root, "without-observation"))
	if err != nil {
		t.Fatal(err)
	}
	gotModel, err := densecausal.Load(filepath.Join(root, "with-observation"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotModel.Weights, wantModel.Weights) {
		t.Fatal("instrumented checkpoint weights differ")
	}
	if len(got.RouterObservations) != 1 {
		t.Fatalf("router observation identities=%d want=1", len(got.RouterObservations))
	}
	coverageTarget, found, err := store.ResolveAlias(t.Context(), runrecord.MoERouterObservationCoverageAlias)
	if err != nil || !found || coverageTarget != got.RouterObservationCoverage {
		t.Fatalf("router coverage alias = (%s, %t, %v)", coverageTarget, found, err)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(t.Context(), store, got.RouterObservationCoverage)
	if err != nil || coverage.Chunk != got.RouterObservations[0] || len(chunk.Observations) != 1 {
		t.Fatalf("router chunk = (coverage=%+v observations=%d err=%v)", coverage, len(chunk.Observations), err)
	}
	router := chunk.Observations[0]
	if router.ID.Kind() != artifact.KindEvidence {
		t.Fatalf("router observation identity = %s", router.ID)
	}
	run, err := runrecord.RequireRun(t.Context(), store, router.Run)
	if err != nil {
		t.Fatal(err)
	}
	code, err := runrecord.NewCodeRevision(executableCommit)
	if err != nil {
		t.Fatal(err)
	}
	session, err := runrecord.RequireServingObservation(t.Context(), store, got.Observation)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(run.Inputs, router.Model) ||
		router.Dataset != got.Checkpoint.Dataset || router.Split != got.Checkpoint.Split ||
		router.Recipe != recipeID || router.Checkpoint != got.Checkpoint.ID() ||
		router.Policy != got.Checkpoint.RunPlan || router.Code != code.ID ||
		session.Run != run.ID || !slices.Contains(run.Outputs, got.Checkpoint.ID()) {
		t.Fatalf("router authority is not exact: router=%+v run=%+v session=%+v", router, run, session)
	}
}

func TestRouterObservationCoverageAndRetentionFailClosed(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("cd", 20))
	incomplete, err := newRouterObservationCollector(0, 1, []int{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	observation := densecausal.MoERouterObservation{
		Layer: 1, Rows: 1, Experts: 2, TopK: 1,
		Selections: []int{0}, CombineWeights: []float32{1}, Accepted: []bool{true},
		Margins: []densecausal.MoERouterMargin{{Observed: true, Value: 1}},
	}
	if err := incomplete.observe(0, observation); err != nil {
		t.Fatal(err)
	}
	if _, _, err := incomplete.complete(); err == nil {
		t.Fatal("missing routed layer accepted")
	}
	if err := incomplete.observe(0, observation); err == nil {
		t.Fatal("duplicate routed layer accepted")
	}
	bounded, err := newRouterObservationCollector(0, 1, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	bounded.encoded = artifact.MaxContentBytes
	if err := bounded.observe(0, observation); err == nil {
		t.Fatal("capture beyond the repository content bound accepted")
	}

	root, model, dataset, store, recipeID := moeWorkflowFixture(t)
	request := Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		Steps: 1, Host: true, Observations: store,
	}
	request.OutputDirectory = filepath.Join(root, "first-retained")
	first, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.OutputDirectory = filepath.Join(root, "second-retained")
	second, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if !first.RouterObservationCoverage.Valid() || !second.RouterObservationCoverage.Valid() ||
		first.RouterObservationCoverage == second.RouterObservationCoverage {
		t.Fatalf("coverage heads = %s/%s", first.RouterObservationCoverage, second.RouterObservationCoverage)
	}
	current, found, err := store.ResolveAlias(t.Context(), runrecord.MoERouterObservationCoverageAlias)
	if err != nil || !found || current != second.RouterObservationCoverage {
		t.Fatalf("current coverage alias = (%s, %t, %v)", current, found, err)
	}
	destination := filepath.Join(root, "compacted-observations")
	if _, err := overgodb.Compact(t.Context(), store, destination, nil); err != nil {
		t.Fatal(err)
	}
	compacted, err := overgodb.OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	if found, err := compacted.HasContent(t.Context(), second.RouterObservationCoverage); err != nil || !found {
		t.Fatalf("current coverage retained = (%t, %v)", found, err)
	}
	for _, id := range second.RouterObservations {
		if found, err := compacted.HasContent(t.Context(), id); err != nil || !found {
			t.Fatalf("current router observation %s retained = (%t, %v)", id, found, err)
		}
	}
	if found, err := compacted.HasContent(t.Context(), first.RouterObservationCoverage); err != nil || found {
		t.Fatalf("superseded coverage retained = (%t, %v)", found, err)
	}
	for _, id := range first.RouterObservations {
		if found, err := compacted.HasContent(t.Context(), id); err != nil || found {
			t.Fatalf("superseded router observation %s retained = (%t, %v)", id, found, err)
		}
	}
}

func TestTrainingRequestCannotOverrideExecutableCodeRevision(t *testing.T) {
	if _, present := reflect.TypeFor[Request]().FieldByName("CodeCommit"); present {
		t.Fatal("training request exposes a caller-controlled code revision")
	}
}

func stubExecutableCodeCommit(t *testing.T, commit string) {
	t.Helper()
	prior := executableCodeCommit
	executableCodeCommit = func(string) (string, error) { return commit, nil }
	t.Cleanup(func() { executableCodeCommit = prior })
}

func moeWorkflowFixture(t *testing.T) (string, string, string, *overgodb.Store, artifact.ID) {
	t.Helper()
	root := t.TempDir()
	spec := testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8,
		KVHeads: 2, Intermediate: 32, Layers: 2, Seed: 17,
		MoELayer: 1, MoEExperts: 4, MoEIntermediate: 8, MoEShared: 8,
	}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	model := filepath.Join(root, "model")
	mixtureConfig := `{"model_type":"deepseek_v2","num_attention_heads":2,"head_dim":8,"max_position_embeddings":16,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true,"num_experts_per_tok":2,"moe_intermediate_size":8,"routed_scaling_factor":1,"norm_topk_prob":true,"scoring_func":"softmax"}`
	writeModelDocument(t, model, weights, shapes, mixtureConfig)
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("abcd"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveTokenPrediction)
	return root, model, dataset, store, recipeID
}

func TestTrainingObjectiveAdmissionMatrix(t *testing.T) {
	policy := testutil.ArtifactID(t, artifact.KindModel, "objective admission policy")
	reference := testutil.ArtifactID(t, artifact.KindModel, "objective admission reference")
	policyDependency := []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: policy}}
	preferenceDependencies := append(append([]recipe.Dependency(nil), policyDependency...), recipe.Dependency{
		Role: recipe.DependencyModel, Slot: 1, Artifact: reference,
	})
	cases := []struct {
		name         string
		want         trainingprogram.ObjectiveKind
		definition   func([]recipe.Dependency) (recipe.Definition, error)
		dependencies []recipe.Dependency
	}{
		{name: "token", want: trainingprogram.ObjectiveTokenPrediction, definition: tokenDefinition, dependencies: policyDependency},
		{name: "dpo", want: trainingprogram.ObjectiveDPO, definition: modelrecipetest.DPODefinition, dependencies: preferenceDependencies},
		{name: "grpo", want: trainingprogram.ObjectiveGRPO, definition: grpoDefinition, dependencies: policyDependency},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			definition, err := test.definition(test.dependencies)
			if err != nil {
				t.Fatal(err)
			}
			program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
			if err != nil {
				t.Fatal(err)
			}
			objective, err := ProgramObjective(program)
			if err != nil || objective != test.want {
				t.Fatalf("objective=(%s,%v), want %s", objective, err, test.want)
			}
		})
	}

	unsupported, err := compositionTrainingDefinition(policyDependency)
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(unsupported, workflowrecipe.Catalog())
	if err != nil {
		t.Fatal(err)
	}
	if objective, err := ProgramObjective(program); err == nil || objective != "" {
		t.Fatalf("unsupported composition objective=(%q,%v), want refusal", objective, err)
	}
}

func TestGRPOUsesSharedRecipeTrainingRuntime(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 9,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "group reward evaluator")
	dataset := filepath.Join(root, "rollouts.jsonl")
	rows := `{"id":"candidate-a","group":"prompt-a","prompt":"ab","completion":"c","reward":1,"evaluator":"` + evaluator.String() + `"}` + "\n" +
		`{"id":"candidate-b","group":"prompt-a","prompt":"ab","completion":"d","reward":-1,"evaluator":"` + evaluator.String() + `"}` + "\n"
	if err := os.WriteFile(dataset, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveGRPO)
	var observed []trainingprogram.GRPOObservation
	result, err := Execute(t.Context(), Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "output"), Steps: 1, Host: true,
		ObjectiveScale: 1, ObserveGRPO: func(value trainingprogram.GRPOObservation) { observed = append(observed, value) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Objective != trainingprogram.ObjectiveGRPO || result.Backend != "host" ||
		result.Checkpoint.ID().Kind() != artifact.KindCheckpoint || len(observed) != 1 ||
		observed[0].Evaluator != evaluator || observed[0].GroupSize != 2 || observed[0].RewardDispersion == 0 {
		t.Fatalf("result=%+v observations=%+v", result, observed)
	}
}

func testTokenResume(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("abcd"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveTokenPrediction)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		Host: true, Observations: store,
	})
}

func TestTrainingObservationSharesDirectedLifecycle(t *testing.T) {
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	observer, err := NewObserver(store, true)
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "observed model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "observed recipe")
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, recipeID)
	if err := observer.Admit(t.Context(), modelID, recipeID); err != nil {
		t.Fatal(err)
	}
	observationID, err := observer.Finish(t.Context(), modelID, recipeID, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.RequireServingObservation(t.Context(), store, observationID); err != nil {
		t.Fatal(err)
	}
}

func testDPOResume(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 7,
	})
	root := t.TempDir()
	policy := filepath.Join(root, "policy")
	reference := filepath.Join(root, "reference")
	writeModel(t, policy, weights, shapes)
	writeModel(t, reference, weights, shapes)
	dataset := filepath.Join(root, "preference.jsonl")
	if err := os.WriteFile(dataset, []byte("{\"id\":\"pair\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, policy, reference, dataset, trainingprogram.ObjectiveDPO)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID,
		ModelDirectory: policy, ReferenceDirectory: reference, DatasetPath: dataset,
		ObjectiveScale: 0.1, Host: true,
	})
}

func testGRPOResume(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 11,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "resume reward evaluator")
	dataset := filepath.Join(root, "rollouts.jsonl")
	rows := `{"id":"candidate-a","group":"prompt-a","prompt":"ab","completion":"c","reward":1,"evaluator":"` + evaluator.String() + `"}` + "\n" +
		`{"id":"candidate-b","group":"prompt-a","prompt":"ab","completion":"d","reward":-1,"evaluator":"` + evaluator.String() + `"}` + "\n"
	if err := os.WriteFile(dataset, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveGRPO)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		ObjectiveScale: 1, Host: true,
	})
}

func verifyWorkflowResume(t *testing.T, root string, request Request) {
	t.Helper()
	request.Steps = 2
	request.OutputDirectory = filepath.Join(root, "uninterrupted")
	want, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.Steps = 1
	request.OutputDirectory = filepath.Join(root, "first")
	first, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.ModelDirectory = ""
	request.ResumeDirectory = request.OutputDirectory
	request.OutputDirectory = filepath.Join(root, "resumed")
	got, err := Execute(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	if request.Observations != nil {
		for _, result := range []Result{want, first, got} {
			if _, err := runrecord.RequireServingObservation(t.Context(), request.Observations, result.Observation); err != nil {
				t.Fatal(err)
			}
		}
	}
	wantModel, err := densecausal.Load(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotModel, err := densecausal.Load(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	wantCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "uninterrupted"))
	if err != nil {
		t.Fatal(err)
	}
	gotCheckpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(root, "resumed"))
	if err != nil {
		t.Fatal(err)
	}
	if want.Backend != "host" || got.Backend != "host" || want.Objective != got.Objective {
		t.Fatalf("execution authority differs: uninterrupted=(%s,%s) resumed=(%s,%s)", want.Backend, want.Objective, got.Backend, got.Objective)
	}
	if first.StreamPosition >= got.StreamPosition || want.StreamPosition != got.StreamPosition {
		t.Fatalf("stream position differs: uninterrupted=%d split=%d+%d", want.StreamPosition, first.StreamPosition, got.StreamPosition)
	}
	if want.Checkpoint.ID() != wantCheckpoint.ID() || got.Checkpoint.ID() != gotCheckpoint.ID() ||
		wantCheckpoint.ID().Kind() != artifact.KindCheckpoint || gotCheckpoint.ID().Kind() != artifact.KindCheckpoint ||
		wantCheckpoint.ID() == gotCheckpoint.ID() {
		t.Fatalf("checkpoint identities do not preserve atomic publication and resume lineage: uninterrupted=%s first=%s resumed=%s", wantCheckpoint.ID(), first.Checkpoint.ID(), gotCheckpoint.ID())
	}
	if !hasLineage(gotCheckpoint.Lineage, first.Checkpoint.ID(), artifact.RelationDerivedFrom) {
		t.Fatalf("resumed checkpoint %s does not derive from split checkpoint %s", gotCheckpoint.ID(), first.Checkpoint.ID())
	}
	if !reflect.DeepEqual(wantModel.Weights, gotModel.Weights) {
		t.Fatal("resumed weights differ from uninterrupted weights")
	}
	checkpointComparisons := []struct {
		name  string
		equal bool
	}{
		{"program", wantCheckpoint.Program == gotCheckpoint.Program},
		{"weights", wantCheckpoint.Weights == gotCheckpoint.Weights},
		{"dataset", wantCheckpoint.Dataset == gotCheckpoint.Dataset},
		{"split", wantCheckpoint.Split == gotCheckpoint.Split},
		{"stream", wantCheckpoint.Stream == gotCheckpoint.Stream},
		{"parameter count", wantCheckpoint.ParameterCount == gotCheckpoint.ParameterCount},
		{"accumulation", wantCheckpoint.Accumulation == gotCheckpoint.Accumulation},
		{"optimizer", reflect.DeepEqual(wantCheckpoint.Optimizer, gotCheckpoint.Optimizer)},
		{"rng", reflect.DeepEqual(wantCheckpoint.RNG, gotCheckpoint.RNG)},
		{"processors", reflect.DeepEqual(wantCheckpoint.Processors, gotCheckpoint.Processors)},
		{"projectors", reflect.DeepEqual(wantCheckpoint.Projectors, gotCheckpoint.Projectors)},
		{"codecs", reflect.DeepEqual(wantCheckpoint.Codecs, gotCheckpoint.Codecs)},
	}
	for _, comparison := range checkpointComparisons {
		if !comparison.equal {
			t.Fatalf("resumed checkpoint %s differs from uninterrupted checkpoint", comparison.name)
		}
	}
	if !reflect.DeepEqual(want.Losses, append(append([]float64(nil), first.Losses...), got.Losses...)) ||
		!reflect.DeepEqual(want.DPO, append(append([]trainingprogram.DPOObservation(nil), first.DPO...), got.DPO...)) ||
		!reflect.DeepEqual(want.GRPO, append(append([]trainingprogram.GRPOObservation(nil), first.GRPO...), got.GRPO...)) {
		t.Fatalf("training observations differ: losses=%d/%d dpo=%d/%d grpo=%d/%d", len(want.Losses), len(first.Losses)+len(got.Losses), len(want.DPO), len(first.DPO)+len(got.DPO), len(want.GRPO), len(first.GRPO)+len(got.GRPO))
	}
}

func hasLineage(lineage []trainingprogram.LineageParent, parent artifact.ID, relation artifact.Relation) bool {
	for _, candidate := range lineage {
		if candidate.Artifact == parent && candidate.Relation == relation {
			return true
		}
	}
	return false
}

func writeModel(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"max_position_embeddings":16,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	writeModelDocument(t, directory, weights, shapes, config)
}

func writeModelDocument(t *testing.T, directory string, weights map[string][]float32, shapes map[string][]int, config string) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	tokenizer := `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(tokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
}

func trainingAuthority(t *testing.T, policyDirectory, referenceDirectory, datasetPath string, objectiveKind trainingprogram.ObjectiveKind) (*overgodb.Store, artifact.ID) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	policy, err := identifyModel(policyDirectory)
	if err != nil {
		t.Fatal(err)
	}
	var reference artifact.ID
	if referenceDirectory != "" {
		reference, err = identifyModel(referenceDirectory)
		if err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	dataset, _ := artifact.IdentifyBytes(artifact.KindDataset, raw)
	split, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	processorName := "text-utf8"
	var evaluators []artifact.ID
	if objectiveKind == trainingprogram.ObjectiveDPO {
		processorName = "preference-token-pair"
	} else if objectiveKind == trainingprogram.ObjectiveGRPO {
		processorName = "grouped-rollout"
		tokenizer, loadErr := hfbpe.Load(policyDirectory)
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		groups, found, parseErr := parseRollouts(raw, tokenizer.Encode)
		_ = groups
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		evaluators = found
	}
	processor, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training/"+processorName+"/v1"))
	profile := func(name string) artifact.ID { return testutil.ArtifactID(t, artifact.KindProfile, name) }
	prefix := string(objectiveKind) + " "
	evaluation := profile(prefix + "evaluation")
	policies := trainingprogram.PolicySpec{
		Precision: profile(prefix + "precision"), Placement: profile(prefix + "placement"), Memory: profile(prefix + "memory"),
		Checkpoint: profile(prefix + "checkpoint"), Evaluation: evaluation, Promotion: profile(prefix + "promotion"),
	}
	optimizerPolicy := trainingprogram.BuiltinOptimizerPolicy()
	policies.Optimizer = optimizerPolicy.ID
	loss, evidence := profile(prefix+"loss"), testutil.ArtifactID(t, artifact.KindEvidence, prefix+"evidence")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: string(objectiveKind), Kind: objectiveKind,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: dataset, Split: split, Processors: []artifact.ID{processor}, Loss: loss,
		Evaluation: evaluation, Metric: trainingprogram.MetricTokenAccuracy,
		Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		t.Fatal(err)
	}
	policies.Objective = objective.ID
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	optimizerContent, err := optimizerPolicy.Content()
	if err != nil {
		t.Fatal(err)
	}
	ids := []artifact.ID{
		policy, dataset, split, processor, loss, evidence, policies.Precision,
		policies.Placement, policies.Memory, policies.Checkpoint, policies.Evaluation, policies.Promotion,
	}
	if reference.Valid() {
		ids = append(ids, reference)
	}
	ids = append(ids, evaluators...)
	descriptors := make([]artifact.Descriptor, len(ids))
	for index, id := range ids {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "training/workflow/authority", Artifacts: descriptors, Contents: []artifact.Content{content, optimizerContent}}); err != nil {
		t.Fatal(err)
	}
	dependencies := append([]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: policy}}, modelrecipetest.PolicyDependencies(policies)...)
	var definition recipe.Definition
	if objectiveKind == trainingprogram.ObjectiveDPO {
		dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyModel, Slot: 1, Artifact: reference})
		definition, err = modelrecipetest.DPODefinition(dependencies)
	} else if objectiveKind == trainingprogram.ObjectiveGRPO {
		for index, evaluator := range evaluators {
			dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyEvaluator, Slot: uint32(index), Artifact: evaluator})
		}
		definition, err = grpoDefinition(dependencies)
	} else {
		definition, err = tokenDefinition(dependencies)
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/workflow/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(ctx, store, "training/workflow/verification", definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "training workflow fixture"); err != nil {
		t.Fatal(err)
	}
	return store, definition.ID
}

func grpoDefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchRollout, Placement: recipe.PlacementHost},
		{ID: "policy", Module: workflowrecipe.ModuleScorePolicy, Placement: recipe.PlacementHost},
		{ID: "objective", Module: workflowrecipe.ModuleGRPOObjective, Placement: recipe.PlacementHost},
		{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
		{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("batch", "batch", "policy", "batch"), edge("policy", "scores", "objective", "policy"),
			edge("objective", "loss", "backward", "loss"), edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
}

func tokenDefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost},
		{ID: "forward", Module: workflowrecipe.ModuleTrainingForward, Placement: recipe.PlacementHost},
		{ID: "backward", Module: workflowrecipe.ModuleBackward, Placement: recipe.PlacementHost},
		{ID: "optimize", Module: workflowrecipe.ModuleOptimize, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("batch", "batch", "forward", "batch"), edge("forward", "loss", "backward", "loss"),
			edge("backward", "gradients", "optimize", "gradients"),
		}, nil,
		[]recipe.Output{{Name: "checkpoint", Data: recipe.DataCheckpoint, Source: recipe.Endpoint{Node: "optimize", Port: "checkpoint"}}},
	)
}

func compositionTrainingDefinition(dependencies []recipe.Dependency) (recipe.Definition, error) {
	nodes := []recipe.Node{
		{ID: "select", Module: workflowrecipe.ModuleSelectComponent, Placement: recipe.PlacementHost},
		{ID: "train", Module: workflowrecipe.ModuleTrainBridge, Placement: recipe.PlacementHost},
		{ID: "evaluate", Module: workflowrecipe.ModuleEvaluateBridge, Placement: recipe.PlacementHost},
	}
	edge := func(fromNode recipe.NodeID, fromPort recipe.PortName, toNode recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{From: recipe.Endpoint{Node: fromNode, Port: fromPort}, To: recipe.Endpoint{Node: toNode, Port: toPort}}
	}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("select", "component", "train", "component"),
			edge("train", "bridge", "evaluate", "bridge"),
		}, nil,
		[]recipe.Output{{Name: "metrics", Data: recipe.DataMetrics, Source: recipe.Endpoint{Node: "evaluate", Port: "metrics"}}},
	)
}
