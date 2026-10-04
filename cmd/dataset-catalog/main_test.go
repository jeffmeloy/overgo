package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingdata"
)

func TestPublishDatasetCatalogCommand(t *testing.T) {
	repository, legacyRoot, contentRoot := t.TempDir(), t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(legacyRoot, "datasets.log"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(contentRoot, "fixture.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	args := []string{"-repo", repository, "-legacy", legacyRoot, "-root", contentRoot}
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	coverage, err := dataset.InspectCatalog(t.Context(), store)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Registered != 1 || coverage.Available != coverage.Registered ||
		!strings.Contains(output.String(), "changed=true") {
		t.Fatalf("coverage=%+v output=%q", coverage, output.String())
	}
}

// TestCatalogPublishesAMixture registers two directory datasets, publishes
// a mixture of them weighted six to two, and holds the materializer to
// reading its normalized weights back: every four records the stream draws
// three from the first and one from the second.
func TestCatalogPublishesAMixture(t *testing.T) {
	repository := t.TempDir()
	corpora := map[string]string{"alpha": "alpha record", "beta": "beta record"}
	for name, text := range corpora {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, name+".txt"), []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := run([]string{"-repo", repository, "-register", name, "-root", root, "-modality", "text"}, io.Discard); err != nil {
			t.Fatal(err)
		}
	}
	var output bytes.Buffer
	if err := run([]string{"-repo", repository, "-mixture", "mix", "-members", "alpha=6,beta=2"}, &output); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-repo", repository, "-mixture", "mix", "-members", "alpha=6,beta=2"}, io.Discard); err != nil {
		t.Fatalf("republishing the same mixture: %v", err)
	}
	if err := run([]string{"-repo", repository, "-mixture", "bad", "-members", "alpha=6,absent=2"}, io.Discard); err == nil {
		t.Fatal("a mixture of an unregistered dataset published")
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	mixture, found, err := dataset.Resolve(ctx, store, dataset.RegisteredAliasPrefix+"mix")
	if err != nil || !found || mixture.Type != dataset.TypeMixture || !strings.Contains(output.String(), mixture.ID.String()) {
		t.Fatalf("registered mixture = %+v, %t, %v; output %q", mixture, found, err, output.String())
	}
	processor, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("mixture-test/text"))
	if err != nil {
		t.Fatal(err)
	}
	text := []recipecontract.Modality{recipecontract.ModalityText}
	materialized, err := trainingdata.MaterializeWhole(ctx, store, trainingdata.Authority{
		Dataset: mixture.ID, Processors: []artifact.ID{processor},
		Signature: recipecontract.ModalitySignature{Inputs: text, Outputs: text},
	}, []trainingdata.ProcessorBinding{{Artifact: processor, Modalities: text,
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, trainingdata.EncodingUTF8)}})
	if err != nil {
		t.Fatal(err)
	}
	defer materialized.Close()
	stream, err := trainingdata.NewStream(materialized, nil)
	if err != nil {
		t.Fatal(err)
	}
	const draws = 8
	batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: draws, DecodeWorkers: 1})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := batcher.Next(ctx)
	if err != nil {
		t.Fatal(err)
	}
	drawn := map[string]int{}
	for _, example := range batch.Examples {
		drawn[string(example.Values[0].Data)]++
	}
	if drawn[corpora["alpha"]] != 6 || drawn[corpora["beta"]] != 2 {
		t.Fatalf("eight draws from the mixture = %v, want six alpha and two beta", drawn)
	}
}
