package trainingdata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
)

// TestInventoryRecordsReadTheirFiles holds a registered directory dataset to
// what training reads from it: each record of its inventory asset is one
// listed file, read whole under the identity dataset.RecordID names, and a
// file whose bytes changed after registration is refused rather than read.
func TestInventoryRecordsReadTheirFiles(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	corpus := t.TempDir()
	contents := map[string]string{"alpha.txt": "alpha text", "beta.txt": "beta text"}
	for name, content := range contents {
		if err := os.WriteFile(filepath.Join(corpus, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registered, err := dataset.RegisterDirectoryDataset(ctx, store, "texts", corpus)
	if err != nil {
		t.Fatal(err)
	}
	source := registered.Dataset
	records := []dataset.Record{
		{ID: dataset.RecordID(source, dataset.InventoryAssetName, 0), Group: "only"},
		{ID: dataset.RecordID(source, dataset.InventoryAssetName, 1), Group: "only"},
	}
	membership, err := dataset.NewMembership(source, 1, "all", records)
	if err != nil {
		t.Fatal(err)
	}
	content, err := membership.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test-membership", Contents: []artifact.Content{content}, Lineage: []artifact.Lineage{membership.Lineage()},
	}); err != nil {
		t.Fatal(err)
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/inventory-asset-test/text"))
	if err != nil {
		t.Fatal(err)
	}
	signature := recipecontract.ModalitySignature{
		Inputs: []recipecontract.Modality{recipecontract.ModalityText}, Outputs: []recipecontract.Modality{recipecontract.ModalityText},
	}
	materialize := func() *Dataset {
		t.Helper()
		materialized, err := Materialize(ctx, store, Authority{
			Dataset: source, Split: membership.ID, Processors: []artifact.ID{processor}, Signature: signature,
		}, []ProcessorBinding{{
			Artifact: processor, Modalities: []recipecontract.Modality{recipecontract.ModalityText},
			Process: Passthrough(RoleInput, recipecontract.ModalityText, "utf-8"),
		}})
		if err != nil {
			t.Fatal(err)
		}
		return materialized
	}
	materialized := materialize()
	var read []string
	for _, member := range materialized.members {
		for _, record := range member.records {
			data, err := record.read()
			if err != nil {
				t.Fatal(err)
			}
			read = append(read, string(data))
		}
	}
	if err := materialized.Close(); err != nil {
		t.Fatal(err)
	}
	if len(read) != len(contents) || !strings.Contains(strings.Join(read, "|"), "alpha text") || !strings.Contains(strings.Join(read, "|"), "beta text") {
		t.Fatalf("inventory records read %q, want each file's bytes", read)
	}

	if err := os.WriteFile(filepath.Join(corpus, "beta.txt"), []byte("beta TEXT"), 0o600); err != nil {
		t.Fatal(err)
	}
	changed := materialize()
	defer changed.Close()
	var refused error
	for _, member := range changed.members {
		for _, record := range member.records {
			if _, err := record.read(); err != nil {
				refused = err
			}
		}
	}
	if refused == nil || !strings.Contains(refused.Error(), "changed since its dataset was registered") {
		t.Fatalf("a file changed after registration read as %v, want it refused", refused)
	}
}
