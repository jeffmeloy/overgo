package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestBenchmarkPublishesUnderTheRunnersModel holds a benchmark's published identity to the model its runner
// measured, the candidate recipe's model, for a model of one file and the same model split in two: a file's own
// bytes identified neither, so publication refused every measured model.
func TestBenchmarkPublishesUnderTheRunnersModel(t *testing.T) {
	ctx := t.Context()
	// fixtureContext: the hermetic model's context, which the identity does not read.
	const fixtureContext = 64
	whole := testutil.HermeticLlamaGGUF(t, fixtureContext)
	source, err := gguf.Open(whole)
	if err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join(t.TempDir(), "split")
	shard := func(index, count uint16) string { return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, index+1, count) }
	// Two shards: a split takes at least one tensor a shard, and the fixture's first shard holds its first tensor.
	err = source.WriteSplit(func(index, count uint16) (io.WriteCloser, error) {
		return os.Create(shard(index, count))
	}, gguf.SplitOptions{MaxTensors: len(source.Tensors) - 1})
	if closeErr := source.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	for name, path := range map[string]string{"whole": whole, "split": shard(0, 2)} {
		candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
		if err != nil {
			t.Fatal(err)
		}
		published, err := publishedModel(path)
		if err != nil || published != candidate.Definition.Model {
			t.Fatalf("%s: published under %s (%v), measured as %s", name, published, err, candidate.Definition.Model)
		}
	}
}
