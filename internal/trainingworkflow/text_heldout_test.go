package trainingworkflow

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
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

// approveTextObjective registers the documents as a dataset, one file per
// record, split in halves by record, under a token-prediction objective
// judged by metric; it trains a tiny dense model through the held-out text
// run and holds the verdict to passing and the objective to climbing to
// approved on it.
func approveTextObjective(t *testing.T, documents []string, metric trainingprogram.EvaluationMetric) {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	modelDirectory := filepath.Join(root, "model")
	writeModel(t, modelDirectory, weights, shapes)
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
	defer store.Close()
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

	result, err := RunTextHeldout(ctx, store, TextHeldoutRequest{Alias: alias, ModelDirectory: modelDirectory, OutputDirectory: filepath.Join(root, "trained"), Steps: 48, Host: true})
	if err != nil || !result.Passed {
		t.Fatalf("text held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	var verdict struct {
		BaselineValue  float64 `json:"baseline_value"`
		CandidateValue float64 `json:"candidate_value"`
	}
	content, found, err := artifact.ReadContent(ctx, store, result.Verdict)
	if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	t.Logf("held-out %s %.4f -> %.4f", metric, verdict.BaselineValue, verdict.CandidateValue)
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}
