package composition

import (
	"context"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestOfflineCompositionArtifactCompatibility(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ctx := context.Background()
	base := testutil.ArtifactID(t, artifact.KindModel, "offline base model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "offline composition recipe")
	compatibility := testutil.ArtifactID(t, artifact.KindProfile, "offline compatibility")
	first := testutil.ArtifactID(t, artifact.KindTensorSet, "offline first task vector")
	second := testutil.ArtifactID(t, artifact.KindTensorSet, "offline second task vector")
	parents := []artifact.ID{base, recipeID, compatibility, first, second}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/offline/parents", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	passthrough, err := NewComposedModel(ComposedModelDocument{
		Architecture: "llama", Recipe: recipeID, Parents: []artifact.ID{base},
		Operation: OfflineCompositionPassthrough, Compatibility: []artifact.ID{compatibility},
	})
	if err != nil {
		t.Fatal(err)
	}
	arithmetic, err := NewComposedModel(ComposedModelDocument{
		Architecture: "llama", Recipe: recipeID, Parents: []artifact.ID{base},
		Components: []artifact.ID{first, second}, Operation: OfflineCompositionTaskArithmetic,
		Compatibility: []artifact.ID{compatibility},
		Arithmetic: []TaskArithmeticTerm{
			{Component: first, Numerator: int64(tensor.SingletonExtent), Denominator: uint64(tensor.SingletonExtent)},
			{Component: second, Numerator: -int64(tensor.SingletonExtent), Denominator: uint64(tensor.PairedExtent)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for key, document := range map[string]ComposedModelDocument{
		"fixture/offline/passthrough": passthrough,
		"fixture/offline/arithmetic":  arithmetic,
	} {
		batch, batchErr := document.Batch(key)
		if batchErr != nil {
			t.Fatal(batchErr)
		}
		if _, commitErr := store.Commit(ctx, batch); commitErr != nil {
			t.Fatal(commitErr)
		}
		edges, parentErr := store.Parents(ctx, document.ID)
		if parentErr != nil {
			t.Fatal(parentErr)
		}
		want := len(document.Lineage())
		if len(edges) != want {
			t.Fatalf("offline lineage edges=%d want=%d: %+v", len(edges), want, edges)
		}
	}
	mismatch := arithmetic
	mismatch.Arithmetic = slices.Clone(arithmetic.Arithmetic)
	mismatch.Arithmetic[tensor.FirstOffset].Component = second
	if _, err := NewComposedModel(mismatch); err == nil {
		t.Fatal("task arithmetic admitted a coefficient bound to the wrong component")
	}
}
