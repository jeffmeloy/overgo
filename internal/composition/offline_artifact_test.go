package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/tensor"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestOfflineCompositionArtifactCompatibility(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())

	base := publishOfflineModel(t, store, "base", offlineTensorWidth, false)
	trained := publishOfflineModel(t, store, "trained", offlineTensorWidth, false)
	passthrough, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactExactPassthrough,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 1}})
	testcheck.NoError(t, err)
	if passthrough.Operator != OfflineArtifactExactPassthrough || len(passthrough.Inputs) != tensor.SingletonExtent ||
		passthrough.ID.Kind() != artifact.KindRecipe || passthrough.Inputs[0].Definition != base {
		t.Fatalf("passthrough plan = %+v", passthrough)
	}

	arithmetic, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactTaskArithmetic,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 0.5}, {Definition: trained, Coefficient: 0.5}})
	testcheck.NoError(t, err)
	content, err := arithmetic.Content()
	testcheck.NoError(t, err)
	if bytes := string(content.Data); strings.Contains(bytes, "bridge") || strings.Contains(bytes, "contract") ||
		strings.Contains(bytes, "injection") || strings.Contains(bytes, "activation") {
		t.Fatalf("offline plan leaked representation-runtime authority: %s", bytes)
	}
	batch, err := arithmetic.Batch("composition/offline/test")
	testcheck.NoError(t, err)
	expectedParents := map[artifact.ID]bool{}
	for _, input := range arithmetic.Inputs {
		expectedParents[input.Model], expectedParents[input.Definition] = true, true
		expectedParents[input.Profile], expectedParents[input.Inventory] = true, true
	}
	if len(batch.Contents) != tensor.SingletonExtent || len(batch.Lineage) != len(expectedParents) {
		t.Fatalf("offline publication = contents %d lineage %d", len(batch.Contents), len(batch.Lineage))
	}

	drifted := publishOfflineModel(t, store, "drifted", offlineTensorWidth+1, false)
	if _, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactTaskArithmetic,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 0.5}, {Definition: drifted, Coefficient: 0.5}}); err == nil ||
		!strings.Contains(err.Error(), "tensor inventories are incompatible") {
		t.Fatalf("inventory drift accepted: %v", err)
	}

	brokenStore := overgodbtest.Open(t, t.TempDir())
	broken := publishOfflineModel(t, brokenStore, "broken-lineage", offlineTensorWidth, true)
	if _, err := CompileOfflineArtifactPlan(ctx, brokenStore, OfflineArtifactExactPassthrough,
		[]OfflineArtifactInput{{Definition: broken, Coefficient: 1}}); err == nil ||
		!strings.Contains(err.Error(), "missing exact lineage") {
		t.Fatalf("missing definition lineage accepted: %v", err)
	}

	if _, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactExactPassthrough,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 0.5}}); err == nil {
		t.Fatal("scaled passthrough accepted")
	}
}

const (
	offlineTensorWidth = uint64(8)
	offlineFloatBytes  = uint64(4)
)

func publishOfflineModel(
	t *testing.T,
	store *overgodb.Store,
	label string,
	width uint64,
	removeDefinitionInventoryLineage bool,
) artifact.ID {
	t.Helper()
	weight := testutil.ArtifactID(t, artifact.KindTensorSet, "offline-weights-"+label)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weight,
	}})
	testcheck.NoError(t, err)
	tensors, err := modelartifact.NewTensorInventoryDocument(
		manifest.ID, modelartifact.TensorFormatSafetensors,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{width, width}, Storage: "f32",
			Bytes: width * width * offlineFloatBytes,
		}},
	)
	testcheck.NoError(t, err)
	profile, found := model.LookupArchitecture("llama")
	if !found {
		t.Fatal("llama profile is absent")
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	testcheck.NoError(t, err)
	spec := model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "llama", BlockCount: 1, ContextLength: 128,
			EmbeddingLength: 8, FeedForwardLength: 16, RMSNormEpsilon: 1e-5,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10_000,
		},
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, tensors, spec)
	testcheck.NoError(t, err)
	resolved, err := document.Resolve(profileDocument, tensors)
	testcheck.NoError(t, err)
	inventory := modelartifact.Inventory{
		Manifest: manifest, TensorInventory: tensors, Components: []artifact.Descriptor{{ID: weight}},
	}
	batch, err := resolved.Batch("composition/offline/fixture/"+label, inventory)
	testcheck.NoError(t, err)
	if removeDefinitionInventoryLineage {
		filtered := batch.Lineage[:0]
		for _, edge := range batch.Lineage {
			if edge.Child != document.ID || edge.Parent != tensors.ID {
				filtered = append(filtered, edge)
			}
		}
		batch.Lineage = filtered
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return document.ID
}
