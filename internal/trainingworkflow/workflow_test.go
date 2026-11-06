package trainingworkflow

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/hfbpe"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
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
	testcheck.NoError(t, os.WriteFile(dataset, []byte("ab"), 0o600))
	store := overgodbtest.Create(t, filepath.Join(root, "repodb"))
	var err error
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "missing active recipe")
	_, err = Execute(t.Context(), Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "output"), Steps: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "not a published training recipe") {
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
	want := testcheck.Must(Execute(t.Context(), base))(t)
	base.OutputDirectory = filepath.Join(root, "with-observation")
	base.Observations = store
	got := testcheck.Must(Execute(t.Context(), base))(t)
	if !reflect.DeepEqual(got.Losses, want.Losses) || got.Checkpoint.ID() != want.Checkpoint.ID() {
		t.Fatalf("instrumented numerics differ: losses=%v/%v checkpoints=%s/%s", got.Losses, want.Losses, got.Checkpoint.ID(), want.Checkpoint.ID())
	}
	wantModel := testcheck.Must(densecausal.Load(filepath.Join(root, "without-observation")))(t)
	gotModel := testcheck.Must(densecausal.Load(filepath.Join(root, "with-observation")))(t)
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
	run := testcheck.Must(runrecord.RequireRun(t.Context(), store, router.Run))(t)
	code := testcheck.Must(runrecord.NewCodeRevision(executableCommit))(t)
	session := testcheck.Must(runrecord.RequireServingObservation(t.Context(), store, got.Observation))(t)
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
	incomplete := testcheck.Must(newRouterObservationCollector(0, 1, []int{1, 3}))(t)
	observation := densecausal.MoERouterObservation{
		Layer: 1, Rows: 1, Experts: 2, TopK: 1,
		Selections: []int{0}, CombineWeights: []float32{1}, Accepted: []bool{true},
		Margins: []densecausal.MoERouterMargin{{Observed: true, Value: 1}},
	}
	testcheck.NoError(t, incomplete.observe(0, observation))
	if _, _, err := incomplete.complete(); err == nil {
		t.Fatal("missing routed layer accepted")
	}
	if err := incomplete.observe(0, observation); err == nil {
		t.Fatal("duplicate routed layer accepted")
	}
	bounded := testcheck.Must(newRouterObservationCollector(0, 1, []int{1}))(t)
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
	first := testcheck.Must(Execute(t.Context(), request))(t)
	request.OutputDirectory = filepath.Join(root, "second-retained")
	second := testcheck.Must(Execute(t.Context(), request))(t)
	if !first.RouterObservationCoverage.Valid() || !second.RouterObservationCoverage.Valid() ||
		first.RouterObservationCoverage == second.RouterObservationCoverage {
		t.Fatalf("coverage heads = %s/%s", first.RouterObservationCoverage, second.RouterObservationCoverage)
	}
	current, found, err := store.ResolveAlias(t.Context(), runrecord.MoERouterObservationCoverageAlias)
	if err != nil || !found || current != second.RouterObservationCoverage {
		t.Fatalf("current coverage alias = (%s, %t, %v)", current, found, err)
	}
	if _, err := overgodb.Release(t.Context(), store, nil, overgodb.RetentionPolicy{}, func(artifact.Descriptor) bool { return true }); err != nil {
		t.Fatal(err)
	}
	if found, err := store.HasContent(t.Context(), second.RouterObservationCoverage); err != nil || !found {
		t.Fatalf("current coverage retained = (%t, %v)", found, err)
	}
	for _, id := range second.RouterObservations {
		if found, err := store.HasContent(t.Context(), id); err != nil || !found {
			t.Fatalf("current router observation %s retained = (%t, %v)", id, found, err)
		}
	}
	if found, err := store.HasContent(t.Context(), first.RouterObservationCoverage); err != nil || found {
		t.Fatalf("superseded coverage retained = (%t, %v)", found, err)
	}
	for _, id := range first.RouterObservations {
		if found, err := store.HasContent(t.Context(), id); err != nil || found {
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
	testcheck.NoError(t, os.WriteFile(dataset, []byte("abcd"), 0o600))
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
			definition := testcheck.Must(test.definition(test.dependencies))(t)
			program := testcheck.Must(recipe.CompileProgram(definition, workflowrecipe.Catalog()))(t)
			objective, err := ProgramObjective(program)
			if err != nil || objective != test.want {
				t.Fatalf("objective=(%s,%v), want %s", objective, err, test.want)
			}
		})
	}

	unsupported := testcheck.Must(compositionTrainingDefinition(policyDependency))(t)
	program := testcheck.Must(recipe.CompileProgram(unsupported, workflowrecipe.Catalog()))(t)
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
	testcheck.NoError(t, os.WriteFile(dataset, []byte(rows), 0o600))
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveGRPO)
	var observed []trainingprogram.GRPOObservation
	result := testcheck.Must(Execute(t.Context(), Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "output"), Steps: 1, Device: hostWeightedStep,
		ObserveGRPO: func(value trainingprogram.GRPOObservation) { observed = append(observed, value) },
	}))(t)
	if result.Objective != trainingprogram.ObjectiveGRPO || result.Backend != "cuda" ||
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
	testcheck.NoError(t, os.WriteFile(dataset, []byte("abcd"), 0o600))
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveTokenPrediction)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		Host: true, Observations: store,
	})
}

// TestDenseProofVerifiesOnItsCheckpointsIdentityAndKeepsNoFile: a dense
// workflow session run as a proof re-verifies its recipe on exactly its
// session's observation and the identity of the checkpoint it trained, as a
// session that is no proof does, and hands its producer the completed gate
// and run. That identity is all the claim reads of the checkpoint, so nothing
// stands at the proof's output, and the store holds the checkpoint's document
// and names no place for it. The same session not run as a proof trains the
// same checkpoint and publishes it at its output, as before. A verification
// binds a named code revision, which this test sets as a package variable, so
// it runs alone.
func TestDenseProofVerifiesOnItsCheckpointsIdentityAndKeepsNoFile(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("f1", 20))
	resultPath := filepath.Join(t.TempDir(), "verification.json")
	t.Setenv(VerificationOutputEnvironment, resultPath)
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	root := t.TempDir()
	model, dataset := filepath.Join(root, "model"), filepath.Join(root, "dataset.txt")
	writeModel(t, model, weights, shapes)
	testcheck.NoError(t, os.WriteFile(dataset, []byte("abcd"), 0o600))
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveTokenPrediction)
	ctx := t.Context()
	request := Request{
		Repository: store, Recipe: recipeID, Observations: store, ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "proof"), Steps: 2, Host: true, Proof: true,
	}
	proof := testcheck.Must(Execute(ctx, request))(t)
	if _, err := os.Stat(request.OutputDirectory); !os.IsNotExist(err) {
		t.Errorf("the proof left its output at %s (%v), want nothing there", request.OutputDirectory, err)
	}
	locations, err := store.Locations(ctx, proof.Checkpoint.ID())
	if _, held, heldErr := artifact.ReadContent(ctx, store, proof.Checkpoint.ID()); err != nil || heldErr != nil || !held || len(locations) != 0 || proof.Candidate.Valid() {
		t.Errorf("the store holds the document of the proof's checkpoint %s: %t, at %v, with candidate %s (%v, %v): want the document, no location and no candidate",
			proof.Checkpoint.ID(), held, locations, proof.Candidate, err, heldErr)
	}
	modelID := testcheck.Must(identifyModel(model))(t)
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskTraining)
	if err != nil || !found {
		t.Fatalf("training activation = (found %t, %v), want the recipe active", found, err)
	}
	verified, err := runrecord.VerifyEvidence(ctx, store, activation.Definition.ID, activation.Event.Evidence)
	if inputs := verified.Run.Inputs; err != nil || len(inputs) != 2 || !slices.Contains(inputs, proof.Observation) || !slices.Contains(inputs, proof.Checkpoint.ID()) {
		t.Fatalf("active verification = (%+v, %v), want it to depend on the proof's session %s and checkpoint %s alone",
			verified.Run, err, proof.Observation, proof.Checkpoint.ID())
	}
	var handoff modelrecipe.Verification
	if err := jsonfile.DecodeStrict(resultPath, &handoff); err != nil || handoff.Gate != verified.Gate.ID || handoff.Run != verified.Run.ID {
		t.Errorf("producer handoff = (%+v, %v), want the proof's completed gate and run", handoff, err)
	}
	// The handoff file must be new for each run that writes one.
	testcheck.NoError(t, os.Remove(resultPath))
	request.Proof, request.OutputDirectory = false, filepath.Join(root, "full")
	full := testcheck.Must(Execute(ctx, request))(t)
	if published, err := trainingprogram.LoadCheckpoint(request.OutputDirectory); err != nil || published.ID() != full.Checkpoint.ID() || full.Checkpoint.ID() != proof.Checkpoint.ID() {
		t.Fatalf("the session that is no proof published checkpoint (%s, %v) at its output and returned %s, want the checkpoint %s the proof identified",
			published.ID(), err, full.Checkpoint.ID(), proof.Checkpoint.ID())
	}
}

func TestTrainingObservationSharesDirectedLifecycle(t *testing.T) {
	store := overgodbtest.Create(t, filepath.Join(t.TempDir(), "repodb"))
	observer := testcheck.Must(NewObserver(store, true))(t)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "observed model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "observed recipe")
	testutil.PublishArtifact(t, store, modelID)
	testutil.PublishArtifact(t, store, recipeID)
	testcheck.NoError(t, observer.Admit(t.Context(), modelID, recipeID))
	observationID := testcheck.Must(observer.Finish(t.Context(), modelID, recipeID, nil, 0))(t)
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
	testcheck.NoError(t, os.WriteFile(dataset, []byte("{\"id\":\"pair\",\"prompt\":\"ab\",\"chosen\":\"c\",\"rejected\":\"d\"}\n"), 0o600))
	store, recipeID := trainingAuthority(t, policy, reference, dataset, trainingprogram.ObjectiveDPO)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID,
		ModelDirectory: policy, ReferenceDirectory: reference, DatasetPath: dataset,
		ObjectiveScale: 0.1, Host: true,
	})
}

func testGRPOResume(t *testing.T) {
	root, request := orderedRewardFixture(t, "c", "d")
	verifyWorkflowResume(t, root, request)
}

// TestRewardSessionPassesATiedGroupAcrossACheckpoint holds a reward session
// to taking no update for a group whose answers all stand together, through
// the checkpoint a session ends in. Of two groups, one with an order and one
// tied, a session of both takes one update and counts one skipped, with the
// stream past both; and the same two units as one session, its checkpoint,
// and a second session resumed from it end at the same weights, the same
// optimizer step and the same place in the stream, however the stream
// orders the two. Those sessions' device trainer is the stand-in that takes
// the weighted step for real on the host (hostWeightedStep). A request that
// names an objective scale is refused for it, held to the host or not: a
// session's inputs are read before where it trains. A session goes to the
// device trainer the request carries, the step fine-tuning takes: each
// answer's completion weighed by its standing, the tied group at zero, the
// stepped group observed and the other counted; and with no device trainer
// it is refused.
func TestRewardSessionPassesATiedGroupAcrossACheckpoint(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 13,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "tied reward evaluator").String()
	dataset := filepath.Join(root, "rollouts.jsonl")
	var rows strings.Builder
	for _, row := range []struct {
		id, group, completion string
		reward                int
	}{{"a", "ordered", "c", 1}, {"b", "ordered", "d", -1}, {"c", "tied", "c", 1}, {"d", "tied", "d", 1}} {
		fmt.Fprintf(&rows, `{"id":"candidate-%s","group":"prompt-%s","prompt":"ab","completion":"%s","reward":%d,"evaluator":"%s"}`+"\n", row.id, row.group, row.completion, row.reward, evaluator)
	}
	testcheck.NoError(t, os.WriteFile(dataset, []byte(rows.String()), 0o600))
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveGRPO)
	request := Request{Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset, Device: hostWeightedStep}
	for _, held := range []bool{false, true} {
		scaled := request
		scaled.Host, scaled.Steps, scaled.ObjectiveScale, scaled.OutputDirectory = held, 2, 1, filepath.Join(root, "scaled")
		if _, err := Execute(t.Context(), scaled); err == nil || !strings.Contains(err.Error(), "no objective scale") {
			t.Fatalf("a reward session with an objective scale, held to the host %t = %v, want it refused for its scale", held, err)
		}
	}
	both := request
	both.Steps, both.OutputDirectory = 2, filepath.Join(root, "both")
	result := testcheck.Must(Execute(t.Context(), both))(t)
	if len(result.GRPO) != 1 || result.Skipped != 1 || result.StreamPosition != 2 || result.Checkpoint.Optimizer.Step != 1 {
		t.Fatalf("a session of an ordered group and a tied one took %d updates to step %d, skipped %d and stands at %d in its stream; want one update, one skipped, and past both",
			len(result.GRPO), result.Checkpoint.Optimizer.Step, result.Skipped, result.StreamPosition)
	}
	verifyWorkflowResume(t, root, request)

	// What the session hands the device's weighted step: each answer one
	// sequence, its completion weighed by its standing, the better answer's
	// loss positively and the worse negatively, and the tied group all zeros.
	// The device here is a stand-in that steps the optimizer once for a group
	// it is to learn from and says what it passed.
	var handed [][]densecausal.WeightedSequence
	device := func(model *densecausal.Model, groups [][]densecausal.WeightedSequence, learningRate, momentum float64, _ bool, resume *densecausal.TrainState, _ densecausal.TrainObserver) ([]float64, int, densecausal.TrainState, error) {
		handed = groups
		update, err := model.NewDenseUpdate(learningRate, momentum, resume)
		if err != nil {
			return nil, 0, densecausal.TrainState{}, err
		}
		var losses []float64
		skipped := 0
		for _, group := range groups {
			if !slices.ContainsFunc(group, densecausal.WeightedSequence.Weighs) {
				skipped++
				continue
			}
			update.Step()
			losses = append(losses, 0.25)
		}
		return losses, skipped, update.Snapshot(), nil
	}
	resident := request
	resident.Device, resident.Steps, resident.OutputDirectory = device, 2, filepath.Join(root, "device")
	onDevice := testcheck.Must(Execute(t.Context(), resident))(t)
	signs := map[bool]int{}
	for _, group := range handed {
		for _, sequence := range group {
			last := sequence.Weights[len(sequence.Weights)-1]
			if len(sequence.Weights) != len(sequence.Tokens)-1 || slices.ContainsFunc(sequence.Weights[:len(sequence.Weights)-1], func(weight float32) bool { return weight != 0 }) {
				t.Fatalf("a reward answer reached the device as %d tokens with weights %v, want a weight on its completion alone", len(sequence.Tokens), sequence.Weights)
			}
			if last != 0 {
				signs[last > 0]++
			}
		}
	}
	if len(handed) != 2 || signs[true] != 1 || signs[false] != 1 {
		t.Fatalf("the device was handed %d groups with %d answers weighed up and %d down, want two groups, one answer each way and the tied group at zero", len(handed), signs[true], signs[false])
	}
	if observed := onDevice.GRPO; onDevice.Backend != "cuda" || onDevice.Skipped != 1 || onDevice.StreamPosition != 2 || onDevice.Checkpoint.Optimizer.Step != 1 || len(observed) != 1 ||
		observed[0].Step != 1 || observed[0].Loss != 0.25 || !(observed[0].LearningRate > 0) || observed[0].GroupSize != 2 || observed[0].CompletionTokens != 2 || observed[0].RewardDispersion != 1 || !observed[0].Valid() {
		t.Fatalf("the reward session on the device = backend %q, %d skipped, position %d, step %d, observations %+v; want one group stepped and observed with its step, rate and rewards, and one skipped",
			onDevice.Backend, onDevice.Skipped, onDevice.StreamPosition, onDevice.Checkpoint.Optimizer.Step, observed)
	}
	resident.Device, resident.OutputDirectory = nil, filepath.Join(root, "no-device")
	if _, err := Execute(t.Context(), resident); err == nil || !strings.Contains(err.Error(), "device trainer") {
		t.Fatalf("a reward session off the host with no device trainer = %v, want it refused", err)
	}
}

// verifyWorkflowResume holds a session stopped at its checkpoint and resumed
// to ending where one uninterrupted session of as many steps ends, and
// returns what the uninterrupted session did.
func verifyWorkflowResume(t *testing.T, root string, request Request, prepareResume ...func(*testing.T, string, trainingprogram.Checkpoint) trainingprogram.Checkpoint) Result {
	t.Helper()
	request.Steps = 2
	request.OutputDirectory = filepath.Join(root, "uninterrupted")
	want := testcheck.Must(Execute(t.Context(), request))(t)
	request.Steps = 1
	request.OutputDirectory = filepath.Join(root, "first")
	first := testcheck.Must(Execute(t.Context(), request))(t)
	for _, prepare := range prepareResume {
		first.Checkpoint = prepare(t, request.OutputDirectory, first.Checkpoint)
	}
	request.ModelDirectory = ""
	request.ResumeDirectory = request.OutputDirectory
	request.OutputDirectory = filepath.Join(root, "resumed")
	got := testcheck.Must(Execute(t.Context(), request))(t)
	if request.Observations != nil {
		for _, result := range []Result{want, first, got} {
			if _, err := runrecord.RequireServingObservation(t.Context(), request.Observations, result.Observation); err != nil {
				t.Fatal(err)
			}
		}
	}
	wantModel := testcheck.Must(loadTrainableModel(filepath.Join(root, "uninterrupted")))(t)
	gotModel := testcheck.Must(loadTrainableModel(filepath.Join(root, "resumed")))(t)
	wantCheckpoint := testcheck.Must(trainingprogram.LoadCheckpoint(filepath.Join(root, "uninterrupted")))(t)
	gotCheckpoint := testcheck.Must(trainingprogram.LoadCheckpoint(filepath.Join(root, "resumed")))(t)
	// A session trains where its request says: on the host, or off it by the device trainer it carries.
	backend := "cuda"
	if request.Host {
		backend = "host"
	}
	if want.Backend != backend || got.Backend != backend || want.Objective != got.Objective {
		t.Fatalf("execution authority differs: uninterrupted=(%s,%s) resumed=(%s,%s), want both on %s", want.Backend, want.Objective, got.Backend, got.Objective, backend)
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
	return want
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
	testcheck.NoError(t, os.MkdirAll(directory, 0o755))
	testcheck.NoError(t, safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil))
	tokenizer := `{"model":{"type":"BPE","vocab":{"a":1,"b":2,"c":3,"d":4},"merges":[]}}`
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600))
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(tokenizer), 0o600))
}

func trainingAuthority(t *testing.T, policyDirectory, referenceDirectory, datasetPath string, objectiveKind trainingprogram.ObjectiveKind) (*overgodb.Store, artifact.ID) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "repodb"))
	testcheck.NoError(t, err)
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	policy := testcheck.Must(identifyModel(policyDirectory))(t)
	var reference artifact.ID
	if referenceDirectory != "" {
		reference = testcheck.Must(identifyModel(referenceDirectory))(t)
	}
	raw := testcheck.Must(os.ReadFile(datasetPath))(t)
	dataset, _ := artifact.IdentifyBytes(artifact.KindDataset, raw)
	split, _ := artifact.IdentifyBytes(artifact.KindDatasetShard, append([]byte("all\x00"), raw...))
	processorName := "text-utf8"
	var evaluators []artifact.ID
	if objectiveKind == trainingprogram.ObjectiveDPO {
		processorName = "preference-token-pair"
	} else if found, prompts := rewardPromptEvaluators(raw); objectiveKind == trainingprogram.ObjectiveGRPO && prompts {
		// A reward session that samples its own answers reads a suite of prompts.
		processorName, evaluators = "reward-prompts", found
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
	objective := testcheck.Must(trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: string(objectiveKind), Kind: objectiveKind,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityText},
		},
		Dataset: dataset, Split: split, Processors: []artifact.ID{processor}, Loss: loss,
		Evaluation: evaluation, Metric: trainingprogram.MetricTokenAccuracy,
		Evidence: []artifact.ID{evidence}, Authority: trainingprogram.ObjectiveApproved,
	}))(t)
	policies.Objective = objective.ID
	content := testcheck.Must(objective.Content())(t)
	optimizerContent := testcheck.Must(optimizerPolicy.Content())(t)
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
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "training/workflow/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "training/workflow/verification", definition.ID))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "training workflow fixture"))
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
	edge := recipe.Connect
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
	edge := recipe.Connect
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
	edge := recipe.Connect
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining, dependencies, nodes,
		[]recipe.Edge{
			edge("select", "component", "train", "component"),
			edge("train", "bridge", "evaluate", "bridge"),
		}, nil,
		[]recipe.Output{{Name: "metrics", Data: recipe.DataMetrics, Source: recipe.Endpoint{Node: "evaluate", Port: "metrics"}}},
	)
}
