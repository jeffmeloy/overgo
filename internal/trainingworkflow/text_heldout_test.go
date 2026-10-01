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
	"overgo/internal/overgodb"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
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
	output := filepath.Join(text.root, "trained")
	result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: steps, MaximumSequence: window, Host: true,
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
		Parameters: muonPlan.ParameterCount(), Optimizer: trainingprogram.BuiltinOptimizerPolicy(),
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
