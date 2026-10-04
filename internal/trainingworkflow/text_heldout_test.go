package trainingworkflow

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/hostoptimizer"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// TestTextObjectiveApprovesOnHeldOutGain trains a tiny dense model under a
// token-prediction objective over a registered dataset of periodic text,
// one file per record: training sees only the training membership, the
// held-out verdict compares next-token accuracy before and after on the
// held-out files, and the verdict climbs the objective to approved.
func TestTextObjectiveApprovesOnHeldOutGain(t *testing.T) {
	var documents []string
	for index := range 12 {
		// Distinct lengths keep every file its own record; the pattern
		// fixes each next token.
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	approveTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
}

// TestGenerationObjectiveApprovesOnHeldOutAnswers trains a tiny dense model
// under a token-prediction objective judged by generated answers: each
// record is a prompt over a and b whose accepted answer is cd after a
// prompt ending in a and dc after one ending in b. Training reads each
// training prompt followed by its answer; the held-out verdict compares
// how many held-out prompts the base and trained models answer, generating
// greedily, and the verdict climbs the objective to approved.
func TestGenerationObjectiveApprovesOnHeldOutAnswers(t *testing.T) {
	var documents []string
	for _, prompt := range []string{"aa", "ab", "ba", "bb", "aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb"} {
		answer := map[byte]string{'a': "cd", 'b': "dc"}[prompt[len(prompt)-1]]
		document, err := json.Marshal(map[string]any{"prompt": prompt, "answers": []string{answer}})
		if err != nil {
			t.Fatal(err)
		}
		documents = append(documents, string(document))
	}
	approveTextObjective(t, documents, trainingprogram.MetricGeneratedAnswer)
}

// TestTextObjectivePacksTrainingRecordsIntoWindows trains under a text
// objective whose records all encode shorter than the run's sequence
// length, each a different length. Training reads windows of exactly that
// length packed from the training records alone, joined by the model's
// end-of-sequence token, which the resident trainer requires; the trained
// model is exactly the base trained on those windows; the held-out records
// stay whole, unpacked; and the held-out verdict publishes.
func TestTextObjectivePacksTrainingRecordsIntoWindows(t *testing.T) {
	const (
		window = 12
		steps  = 9
	)
	var documents []string
	for index := range 10 {
		documents = append(documents, strings.Repeat("abcd", 3)[:2+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	weights, err := IdentifyModelWeights(text.model)
	if err != nil {
		t.Fatal(err)
	}
	defaultRecipe, err := BootstrapObjectiveRecipe(ctx, text.store, weights, text.alias)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.RequireDefinition(ctx, text.store, defaultRecipe)
	if err != nil {
		t.Fatal(err)
	}
	// A declared alternate policy must change the updates themselves, not just the recorded recipe label.
	policy := trainingprogram.BuiltinOptimizerPolicy()
	policy.ParameterExponent, policy.MomentumEffectiveSamples = -0.5, 7
	policy.ID, err = artifact.JSONID(artifact.KindProfile, policy)
	if err != nil {
		t.Fatal(err)
	}
	content, err := policy.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := text.store.Commit(ctx, artifact.Batch{Key: "selected-policy", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	for index := range definition.Dependencies {
		if definition.Dependencies[index].Role == recipe.DependencyOptimizer {
			definition.Dependencies[index].Artifact = policy.ID
		}
	}
	selected, err := recipe.NewDefinitionWithDependencies(definition.Task, definition.Dependencies, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, text.store, "selected-recipe", selected); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(text.root, "trained")
	result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, Recipe: selected.ID, ModelDirectory: text.model, OutputDirectory: output, Steps: steps, MaximumSequence: window, Host: true,
	})
	if err != nil || !result.Verdict.Valid() || !result.Baseline.Valid() || !result.Candidate.Valid() {
		t.Fatalf("text held-out run = (%+v, %v), want a published verdict", result, err)
	}

	tokenizer, err := loadTrainableTokenizer(text.model)
	if err != nil || !tokenizer.ended {
		t.Fatalf("tokenizer = (ended %t, %v), want the declared end-of-sequence token", tokenizer.ended, err)
	}
	data, records, err := textObjectiveData(ctx, text.store, text.objective, tokenizer, window)
	if err != nil {
		t.Fatal(err)
	}
	training, heldout, err := objectiveMemberships(ctx, text.store, text.objective)
	if err != nil {
		t.Fatal(err)
	}
	var packed [][]int
	trained := map[string]bool{}
	for _, record := range training.Records {
		if length := len(records[record.ID].tokens); length >= window {
			t.Fatalf("training record %s encodes to %d tokens, want fewer than the %d-token window", record.ID, length, window)
		}
		packed, trained[record.ID] = append(packed, records[record.ID].tokens), true
	}
	windows, position, err := packWindows(packed, tokenizer.end, window, steps)
	if err != nil {
		t.Fatal(err)
	}
	for step, tokens := range windows {
		if len(tokens) != window {
			t.Fatalf("window %d holds %d tokens, want exactly %d", step, len(tokens), window)
		}
	}
	if position <= uint64(len(packed)) {
		t.Fatalf("%d windows reached %d of %d training records, want them to cycle past the membership", steps, position, len(packed))
	}
	// The held-out records are judged whole, each its own sequence.
	for _, record := range heldout.Records {
		example, tokens := data.Examples[record.ID], records[record.ID].tokens
		if trained[record.ID] || !slices.Equal(dtype.ConvertSlice[int](example.Input), tokens) || !slices.Equal(dtype.ConvertSlice[int](example.Target), tokens[1:]) {
			t.Fatalf("held-out record %s example = %v -> %v, want its whole %v", record.ID, example.Input, example.Target, tokens)
		}
	}

	// The published model is the base trained on exactly these windows.
	base, err := loadTrainableModel(text.model)
	if err != nil {
		t.Fatal(err)
	}
	muonPlan, err := base.TrainingPlan()
	if err != nil {
		t.Fatal(err)
	}
	plan, err := trainingprogram.CompileProbeSessionPlan(trainingprogram.ProbeSpec{
		Objective: trainingprogram.ObjectiveTokenPrediction, Updates: steps, MaximumSequence: window,
		Parameters: muonPlan.ParameterCount(), Optimizer: policy,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := base.Train(windows, plan.Optimizer().BaseLearningRate, plan.Optimizer().Momentum, nil, nil); err != nil {
		t.Fatal(err)
	}
	published, err := loadTrainableModel(output)
	if err != nil {
		t.Fatal(err)
	}
	for name, values := range base.Weights {
		if !slices.Equal(published.Weights[name], values) {
			t.Fatalf("published %s differs from the base trained on the packed windows", name)
		}
	}
}

// TestTextObjectiveReportsEachStep: a held-out text run reports its
// progress as the dense workflow does, the lane it trains on and then, as
// each update completes, the step, its loss, wall and lane.
func TestTextObjectiveReportsEachStep(t *testing.T) {
	t.Parallel()
	const steps = 5
	var documents []string
	for index := range 8 {
		documents = append(documents, strings.Repeat("abcd", 3)[:4+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	var progress strings.Builder
	if _, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "trained"),
		Steps: steps, MaximumSequence: 8, Host: true, Progress: &progress,
	}); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(progress.String(), "\n"), "\n")
	if want := fmt.Sprintf("lane host-reference: %d steps, %d batches, frozen_lexical=false", steps, steps); len(lines) != steps+1 || lines[0] != want {
		t.Fatalf("progress = %q, want %q and a line per step", progress.String(), want)
	}
	for index, line := range lines[1:] {
		var step, of int
		var loss float64
		var wall, backend string
		if _, err := fmt.Sscanf(line, "step %d/%d loss %g wall %s backend %s", &step, &of, &loss, &wall, &backend); err != nil ||
			step != index+1 || of != steps || !(loss > 0) || backend != "host-reference" {
			t.Fatalf("progress line %d = %q (%v), want step %d/%d with its loss, wall and lane", index, line, err, index+1, steps)
		}
	}
}

// TestTextObjectiveSessionRecordsItsLoad: a held-out text run loads its
// model, tokenizer and records before its session opens, and the session
// records how long that took as its one phase, the load, as the dense
// workflow's session records its own; the recipe's load time reads it.
func TestTextObjectiveSessionRecordsItsLoad(t *testing.T) {
	t.Parallel()
	var documents []string
	for index := range 8 {
		documents = append(documents, strings.Repeat("abcd", 3)[:4+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "trained"),
		Steps: 2, MaximumSequence: 8, Host: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	session, err := runrecord.RequireServingObservation(ctx, text.store, result.Observation)
	if err != nil || len(session.Phases) != 1 || session.Phases[0].Phase != runrecord.PhaseLoad || session.Phases[0].DurationNS == 0 {
		t.Fatalf("session phases = (%+v, %v), want the run's load alone", session.Phases, err)
	}
}

// TestFrozenLexicalTextRunRecordsItsFrozenPlan: a held-out text run that
// froze the lexical table records the Muon plan it stepped, the model's
// plan with exactly the embedding and head frozen, so its program, run plan
// and checkpoint, which the verdict binds, differ from an unfrozen run's
// under the same recipe. One host-trained state publishes both ways, so no
// device runs.
func TestFrozenLexicalTextRunRecordsItsFrozenPlan(t *testing.T) {
	t.Parallel()
	var documents []string
	for index := range 8 {
		documents = append(documents, strings.Repeat("abcd", 3)[:4+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	weights, err := IdentifyModelWeights(text.model)
	if err != nil {
		t.Fatal(err)
	}
	recipeID, err := BootstrapObjectiveRecipe(ctx, text.store, weights, text.alias)
	if err != nil {
		t.Fatal(err)
	}
	model, err := loadTrainableModel(text.model)
	if err != nil {
		t.Fatal(err)
	}
	tokenizer, err := loadTrainableTokenizer(text.model)
	if err != nil {
		t.Fatal(err)
	}
	data, _, err := textObjectiveData(ctx, text.store, text.objective, tokenizer, 8)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := model.TrainingPlan()
	if err != nil {
		t.Fatal(err)
	}
	_, state, err := model.Train([][]int{{1, 2, 3, 4}, {4, 3, 2, 1}}, 0.01, 0.9, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	publish := func(frozen bool) trainingprogram.Checkpoint {
		request := TextHeldoutRequest{ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, fmt.Sprintf("trained-%t", frozen)), FreezeLexical: frozen}
		stream := trainingdata.StreamState{Identity: data.Training, Position: 1}
		checkpoint, err := publishTrainedText(ctx, text.store, request, model, plan, recipeID, text.objective, stream, state, nil)
		if err != nil {
			t.Fatal(err)
		}
		return checkpoint
	}
	unfrozen, frozen := publish(false), publish(true)
	embedding, _, head := model.LexicalTensorNames()
	specs := make([]hostoptimizer.GroupSpec, plan.GroupCount())
	for index, group := range plan.Groups() {
		specs[index] = group.GroupSpec
		specs[index].Frozen = group.Name == embedding || group.Name == head
	}
	stepped, err := hostoptimizer.CompilePlan(plan.ParameterCount(), specs)
	if err != nil {
		t.Fatal(err)
	}
	if frozen.Optimizer.PlanIdentity != stepped.Identity() || unfrozen.Optimizer.PlanIdentity != plan.Identity() {
		t.Fatalf("recorded plans = (frozen %s, unfrozen %s), want (%s, %s)", frozen.Optimizer.PlanIdentity, unfrozen.Optimizer.PlanIdentity, stepped.Identity(), plan.Identity())
	}
	if frozen.Program == unfrozen.Program || frozen.RunPlan == unfrozen.RunPlan || frozen.ID() == unfrozen.ID() {
		t.Fatalf("frozen and unfrozen runs recorded one identity: program %s, run plan %s, checkpoint %s", frozen.Program, frozen.RunPlan, frozen.ID())
	}
}

// TestTextObjectiveProofReverifiesWithoutAVerdict: a proof run under a text
// objective re-verifies the recipe a full run activated by training alone.
// It records its session and publishes its checkpoint at its output, and the
// recipe's verification depends on exactly those; it scores neither model,
// so no held-out verdict publishes and the full run's stays the newest, the
// one the recipe's held-out quality reads. A verification binds a named code
// revision, which this test sets as a package variable, so it runs alone.
func TestTextObjectiveProofReverifiesWithoutAVerdict(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("e7", 20))
	resultPath := filepath.Join(t.TempDir(), "verification.json")
	t.Setenv("OVERGO_TRAIN_VERIFICATION", resultPath)
	var documents []string
	for index := range 8 {
		documents = append(documents, strings.Repeat("abcd", 3)[:4+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	run := func(output string, proof bool) HeldoutResult {
		t.Helper()
		result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, output),
			Steps: 2, MaximumSequence: 8, Host: true, Proof: proof,
		})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	full := run("full", false)
	if err := os.Remove(resultPath); err != nil {
		t.Errorf("full run handoff: %v", err)
	}
	proof := run("proof", true)
	if !full.Verdict.Valid() || !proof.Observation.Valid() || proof.Verdict.Valid() || proof.View.Valid() || proof.Plan.Valid() ||
		proof.Baseline.Valid() || proof.Candidate.Valid() {
		t.Fatalf("full run = %+v, proof run = %+v: want a session and checkpoint from each and only the full run judged", full, proof)
	}
	if checkpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(text.root, "proof")); err != nil || checkpoint.ID() != proof.Trained {
		t.Fatalf("proof output holds checkpoint (%s, %v), want %s", checkpoint.ID(), err, proof.Trained)
	}
	var judged []artifact.ID
	if err := evaluation.VisitHeldoutVerdicts(ctx, text.store, func(observation artifact.ID, _, _ evaluation.EvalSummary) error {
		judged = append(judged, observation)
		return nil
	}); err != nil || !slices.Equal(judged, []artifact.ID{full.Observation}) {
		t.Fatalf("held-out verdicts judged sessions (%v, %v), want the full run's %s alone", judged, err, full.Observation)
	}
	modelID, err := identifyModel(text.model)
	if err != nil {
		t.Fatal(err)
	}
	activation, found, err := modelrecipe.ActiveRecord(ctx, text.store, modelID, recipe.TaskTraining)
	if err != nil || !found {
		t.Fatalf("training activation = (found %t, %v), want the recipe active", found, err)
	}
	verified, err := runrecord.VerifyEvidence(ctx, text.store, activation.Definition.ID, activation.Event.Evidence)
	if inputs := verified.Run.Inputs; err != nil || len(inputs) != 2 || !slices.Contains(inputs, proof.Observation) || !slices.Contains(inputs, proof.Trained) {
		t.Fatalf("active verification = (%+v, %v), want it to depend on the proof's session %s and checkpoint %s alone",
			verified.Run, err, proof.Observation, proof.Trained)
	}
	var handoff modelrecipe.Verification
	if err := jsonfile.DecodeStrict(resultPath, &handoff); err != nil || handoff.Gate != verified.Gate.ID || handoff.Run != verified.Run.ID {
		t.Errorf("producer handoff = (%+v, %v), want exact completed proof", handoff, err)
	}
	// Promotion retargets the objective alias. A previously selected recipe must not silently train under the replacement.
	if _, err := PromoteObjectiveAdaptive(ctx, text.store, text.alias, []artifact.ID{full.Observation}); err != nil {
		t.Fatal(err)
	}
	_, err = RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, Recipe: activation.Definition.ID, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "retargeted"),
		Steps: 2, MaximumSequence: 8, Host: true, Proof: true,
	})
	if err == nil || !strings.Contains(err.Error(), "objective differs") {
		t.Fatalf("selected recipe after objective retarget = %v, want objective mismatch refused", err)
	}
}

// TestPackWindowsCyclesAShortStream packs a stream shorter than one window:
// every window is still exactly the window's length, continuing from the
// stream's start past the last document's end token, and the position
// counts each document a window began.
func TestPackWindowsCyclesAShortStream(t *testing.T) {
	const end = 9
	windows, position, err := packWindows([][]int{{1, 2}, {3}}, end, 8, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{{1, 2, end, 3, end, 1, 2, end}, {3, end, 1, 2, end, 3, end, 1}, {2, end, 3, end, 1, 2, end, 3}}
	// 24 tokens read cycle the 5-token stream 4 times, then begin both
	// documents of the fifth pass.
	if !slices.EqualFunc(windows, want, slices.Equal) || position != 10 {
		t.Fatalf("packed = (%v, position %d), want (%v, position 10)", windows, position, want)
	}
	if _, _, err := packWindows(nil, end, 8, 3); err == nil {
		t.Fatal("an empty training membership packed")
	}
}

// approveTextObjective registers the documents as a text objective judged
// by metric, trains a tiny dense model through the held-out text run and
// holds the verdict to passing and the objective to climbing to approved
// on it.
func approveTextObjective(t *testing.T, documents []string, metric trainingprogram.EvaluationMetric) {
	t.Helper()
	text := registerTextObjective(t, documents, metric)
	ctx := t.Context()
	result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "trained"), Steps: 48, Host: true})
	if err != nil || !result.Passed {
		t.Fatalf("text held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	var verdict struct {
		BaselineValue  float64 `json:"baseline_value"`
		CandidateValue float64 `json:"candidate_value"`
	}
	content, found, err := artifact.ReadContent(ctx, text.store, result.Verdict)
	if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	t.Logf("held-out %s %.4f -> %.4f", metric, verdict.BaselineValue, verdict.CandidateValue)
	if _, err := PromoteObjectiveAdaptive(ctx, text.store, text.alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := PromoteObjectiveApproved(ctx, text.store, text.alias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}

// registeredText is a text objective a test registered: the store, the
// test's root, the tiny model's directory, the objective and its alias.
type registeredText struct {
	store     *overgodb.Store
	root      string
	model     string
	objective trainingprogram.ObjectiveDocument
	alias     string
}

// registerTextObjective writes a tiny dense model that declares token zero
// its end of sequence, registers the documents as a dataset, one file per
// record, split in halves by record, and registers a token-prediction
// objective over it judged by metric.
func registerTextObjective(t *testing.T, documents []string, metric trainingprogram.EvaluationMetric) registeredText {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	modelDirectory := filepath.Join(root, "model")
	writeModel(t, modelDirectory, weights, shapes)
	if err := os.WriteFile(filepath.Join(modelDirectory, "generation_config.json"), []byte(`{"eos_token_id":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	corpus := filepath.Join(root, "corpus")
	if err := os.MkdirAll(corpus, 0o755); err != nil {
		t.Fatal(err)
	}
	for index, document := range documents {
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("text-%02d.txt", index)), []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, "texts", corpus)
	if err != nil {
		t.Fatal(err)
	}
	var records []dataset.Record
	for index := range uint64(len(documents)) {
		id := dataset.RecordID(registered.Dataset, dataset.InventoryAssetName, index)
		records = append(records, dataset.Record{ID: id, Group: id})
	}
	split, err := dataset.BuildGroupSplit(registered.Dataset, records, 17, []dataset.SplitPartition{{Name: "heldout", Weight: 1}, {Name: "train", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	splitBatch, err := split.PublicationBatch("test-split", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, splitBatch); err != nil {
		t.Fatal(err)
	}
	var training artifact.ID
	for _, membership := range split.Memberships {
		if membership.Partition == "train" {
			training = membership.ID
		}
	}
	derived := derivedIdentity(t, "overgo/text-heldout-test/")
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "texts", Kind: trainingprogram.ObjectiveTokenPrediction, Signature: textSignature,
		Dataset: registered.Dataset, Split: training, Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss: derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: metric, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.texts"
	registerObjective(t, ctx, store, root, objective, alias)
	return registeredText{store: store, root: root, model: modelDirectory, objective: objective, alias: alias}
}
