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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipecontract"
	"overgo/internal/testcheck"
	"overgo/internal/trainingdata"
)

func TestPublishDatasetCatalogCommand(t *testing.T) {
	repository, legacyRoot, contentRoot := t.TempDir(), t.TempDir(), t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(legacyRoot, "datasets.log"), nil, 0o600))
	testcheck.NoError(t, os.WriteFile(filepath.Join(contentRoot, "fixture.txt"), []byte("x"), 0o600))
	var output bytes.Buffer
	args := []string{"-repo", repository, "-legacy", legacyRoot, "-root", contentRoot}
	testcheck.NoError(t, run(args, &output))
	store := overgodbtest.OpenReadOnly(t, repository)
	coverage := testcheck.Must(dataset.InspectCatalog(t.Context(), store))(t)
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
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, name+".txt"), []byte(text), 0o600))
		testcheck.NoError(t, run([]string{"-repo", repository, "-register", name, "-root", root, "-modality", "text"}, io.Discard))
	}
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository, "-mixture", "mix", "-members", "alpha=6,beta=2"}, &output))
	if err := run([]string{"-repo", repository, "-mixture", "mix", "-members", "alpha=6,beta=2"}, io.Discard); err != nil {
		t.Fatalf("republishing the same mixture: %v", err)
	}
	if err := run([]string{"-repo", repository, "-mixture", "bad", "-members", "alpha=6,absent=2"}, io.Discard); err == nil {
		t.Fatal("a mixture of an unregistered dataset published")
	}
	store := overgodbtest.OpenReadOnly(t, repository)
	ctx := t.Context()
	mixture, found, err := dataset.Resolve(ctx, store, dataset.RegisteredAliasPrefix+"mix")
	if err != nil || !found || mixture.Type != dataset.TypeMixture || !strings.Contains(output.String(), mixture.ID.String()) {
		t.Fatalf("registered mixture = %+v, %t, %v; output %q", mixture, found, err, output.String())
	}
	processor := testcheck.Must(artifact.IdentifyBytes(artifact.KindProfile, []byte("mixture-test/text")))(t)
	text := []recipecontract.Modality{recipecontract.ModalityText}
	materialized := testcheck.Must(trainingdata.MaterializeWhole(ctx, store, trainingdata.Authority{
		Dataset: mixture.ID, Processors: []artifact.ID{processor},
		Signature: recipecontract.ModalitySignature{Inputs: text, Outputs: text},
	}, []trainingdata.ProcessorBinding{{Artifact: processor, Modalities: text,
		Process: trainingdata.Passthrough(trainingdata.RoleInput, recipecontract.ModalityText, trainingdata.EncodingUTF8)}}))(t)
	defer materialized.Close()
	stream := testcheck.Must(trainingdata.NewStream(materialized, nil))(t)
	const draws = 8
	batcher := testcheck.Must(trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: draws, DecodeWorkers: 1}))(t)
	batch := testcheck.Must(batcher.Next(ctx))(t)
	drawn := map[string]int{}
	for _, example := range batch.Examples {
		drawn[string(example.Values[0].Data)]++
	}
	if drawn[corpora["alpha"]] != 6 || drawn[corpora["beta"]] != 2 {
		t.Fatalf("eight draws from the mixture = %v, want six alpha and two beta", drawn)
	}
}
