package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestOfflineCompositionArtifactCompatibility(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	base := publishOfflineModel(t, store, "base", offlineTensorWidth, false)
	trained := publishOfflineModel(t, store, "trained", offlineTensorWidth, false)
	passthrough, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactExactPassthrough,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if passthrough.Operator != OfflineArtifactExactPassthrough || len(passthrough.Inputs) != tensor.SingletonExtent ||
		passthrough.ID.Kind() != artifact.KindRecipe || passthrough.Inputs[0].Definition != base {
		t.Fatalf("passthrough plan = %+v", passthrough)
	}

	arithmetic, err := CompileOfflineArtifactPlan(ctx, store, OfflineArtifactTaskArithmetic,
		[]OfflineArtifactInput{{Definition: base, Coefficient: 0.5}, {Definition: trained, Coefficient: 0.5}})
	if err != nil {
		t.Fatal(err)
	}
	content, err := arithmetic.Content()
	if err != nil {
		t.Fatal(err)
	}
	if bytes := string(content.Data); strings.Contains(bytes, "bridge") || strings.Contains(bytes, "contract") ||
		strings.Contains(bytes, "injection") || strings.Contains(bytes, "activation") {
		t.Fatalf("offline plan leaked representation-runtime authority: %s", bytes)
	}
	batch, err := arithmetic.Batch("composition/offline/test")
	if err != nil {
		t.Fatal(err)
	}
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

	brokenStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer brokenStore.Close()
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
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		manifest.ID, modelartifact.TensorFormatSafetensors,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{width, width}, Storage: "f32",
			Bytes: width * width * offlineFloatBytes,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	profile, found := model.LookupArchitecture("llama")
	if !found {
		t.Fatal("llama profile is absent")
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := document.Resolve(profileDocument, tensors)
	if err != nil {
		t.Fatal(err)
	}
	inventory := modelartifact.Inventory{
		Manifest: manifest, TensorInventory: tensors, Components: []artifact.Descriptor{{ID: weight}},
	}
	batch, err := resolved.Batch("composition/offline/fixture/"+label, inventory)
	if err != nil {
		t.Fatal(err)
	}
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
