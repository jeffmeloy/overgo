package trainingworkflow

import (
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
	const files = 12
	for index := range files {
		// Distinct lengths keep every file its own record; the pattern
		// fixes each next token.
		text := strings.Repeat("abcd", 5)[:6+index]
		if err := os.WriteFile(filepath.Join(corpus, fmt.Sprintf("text-%02d.txt", index)), []byte(text), 0o600); err != nil {
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
	for index := range uint64(files) {
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
		Metric: trainingprogram.MetricTokenAccuracy, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.texts"
	registerObjective(t, ctx, store, root, objective, alias)

	result, err := RunTextHeldout(ctx, store, TextHeldoutRequest{Alias: alias, ModelDirectory: modelDirectory, Steps: 48, Host: true})
	if err != nil || !result.Passed {
		t.Fatalf("text held-out run = (passed=%t, %v), want a passed verdict", result.Passed, err)
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{result.Observation}); err != nil {
		t.Fatal(err)
	}
	approved, err := PromoteObjectiveApproved(ctx, store, alias, []artifact.ID{result.Verdict})
	if err != nil || approved.Authority != trainingprogram.ObjectiveApproved {
		t.Fatalf("approval = (%q, %v), want approved on the held-out verdict", approved.Authority, err)
	}
}
