package modelrecipe

import (
	"bytes"
	"errors"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const (
	definitionArchitecture = "llama"
	definitionEmbedding    = uint32(8)
	definitionContext      = uint32(128)
	definitionFeedForward  = uint32(16)
	definitionHeads        = uint32(2)
	definitionKVHeads      = uint32(1)
	definitionHeadWidth    = uint32(4)
	definitionRopeBase     = float32(10_000)
	definitionNormEpsilon  = float32(1e-5)
	definitionBlockCount   = uint32(1)
	definitionTensorWidth  = uint64(definitionEmbedding)
	definitionF32Bytes     = uint64(4)
	definitionTensorBytes  = definitionTensorWidth * definitionTensorWidth * definitionF32Bytes
)

func TestModelDefinitionRoundTripAndExactProfileCompile(t *testing.T) {
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profile.LayerTopology = model.LayerTopologyCausalPostQKNormSkip
	profileDocument := testcheck.Must(NewProfileDocument(profile))(t)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model-definition-fixture")
	tensors := testcheck.Must(modelartifact.NewTensorInventoryDocument(
		modelID, modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth},
			Storage: "f32", Bytes: definitionTensorBytes,
		}},
	))(t)
	document := testcheck.Must(NewModelDefinitionDocument(profileDocument, tensors, definitionSpec()))(t)
	content := testcheck.Must(document.Content())(t)
	parsed := testcheck.Must(modelDefinitionCodec.Parse(content.Data))(t)
	resolved := testcheck.Must(parsed.Resolve(profileDocument, tensors))(t)
	definition := testcheck.Must(InferenceWithModelDefinition(
		modelID, profileDocument.ID, document.ID, recipe.PlacementHost, DecodeSessionRequest,
		recipe.ResidencyHostCache,
	))(t)
	plan := testcheck.Must(CompileModelDefinition(definition, resolved, model.Weights{}))(t)
	if parsed.ID != document.ID || plan.Model.Profile().LayerTopology != model.LayerTopologyCausalPostQKNormSkip {
		t.Fatalf("resolved definition/plan = (%s, %+v)", parsed.ID, plan.Model.Profile())
	}
}

// TestGGUFModelDefinitionOvergoDBResolution: one model written at two paths
// has one definition, and its facts publish from each path. The second
// publication changes only where the bytes lie, which a batch keyed by the
// definition alone refused as a conflict; the store then resolves the
// definition by identity and holds the new location.
func TestGGUFModelDefinitionOvergoDBResolution(t *testing.T) {
	t.Parallel()
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profileDocument := testcheck.Must(NewProfileDocument(profile))(t)
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	var definition artifact.ID
	for _, path := range []string{filepath.Join(t.TempDir(), "definition.gguf"), filepath.Join(t.TempDir(), "definition.gguf")} {
		writeDefinitionGGUF(t, path)
		file := testcheck.Must(gguf.Open(path))(t)
		inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
		if err != nil {
			t.Fatal(errors.Join(err, file.Close()))
		}
		spec, err := model.ReadSpecWithProfile(file, profile)
		testcheck.NoError(t, errors.Join(err, file.Close()))
		document := testcheck.Must(NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec))(t)
		if definition.Valid() && document.ID != definition {
			t.Fatalf("relocated model definitions differ: %s != %s", document.ID, definition)
		}
		definition = document.ID
		resolvedSource := testcheck.Must(document.Resolve(profileDocument, inventory.TensorInventory))(t)
		if err := PublishResolvedModelDefinition(ctx, store, inventory, resolvedSource); err != nil {
			t.Fatalf("publish from %s: %v", path, err)
		}
		resolved := testcheck.Must(ResolveModelDefinition(ctx, store, document.ID))(t)
		if resolved.Document.Model != inventory.Manifest.ID || resolved.Spec.EmbeddingLength != definitionEmbedding ||
			resolved.Tensors.ID != inventory.TensorInventory.ID {
			t.Fatalf("resolved model definition = %+v", resolved)
		}
		for _, location := range inventory.Locations {
			held, err := store.Locations(ctx, location.Artifact)
			if err != nil || !slices.Contains(held, location) {
				t.Fatalf("store holds %v for %s, want %v: %v", held, location.Artifact, location, err)
			}
		}
	}
}

func TestPublishResolvedModelDefinitionSeparatesProfiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "definition.gguf")
	writeDefinitionGGUF(t, path)
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	base, _ := model.LookupArchitecture(definitionArchitecture)
	spec := testcheck.Must(model.ReadSpecWithProfile(file, base))(t)
	profiles := []model.ArchitectureProfile{base, base}
	profiles[1].LayerTopology = model.LayerTopologyCausalPostQKNormSkip
	store := overgodbtest.Open(t, t.TempDir())
	for _, profile := range profiles {
		profileDocument := testcheck.Must(NewProfileDocument(profile))(t)
		document := testcheck.Must(NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec))(t)
		resolved := testcheck.Must(document.Resolve(profileDocument, inventory.TensorInventory))(t)
		testcheck.NoError(t, PublishResolvedModelDefinition(t.Context(), store, inventory, resolved))
		if _, err := ResolveModelDefinition(t.Context(), store, document.ID); err != nil {
			t.Fatal(err)
		}
	}
}

func TestModelDefinitionRejectsBindingDrift(t *testing.T) {
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profileDocument := testcheck.Must(NewProfileDocument(profile))(t)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "binding-drift-model")
	tensors := testcheck.Must(modelartifact.NewTensorInventoryDocument(
		modelID, modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{Name: "weight", Shape: []uint64{}, Storage: "f32", Bytes: definitionF32Bytes}},
	))(t)
	document := testcheck.Must(NewModelDefinitionDocument(profileDocument, tensors, definitionSpec()))(t)
	otherProfile, _ := model.LookupArchitecture("qwen3")
	otherDocument := testcheck.Must(NewProfileDocument(otherProfile))(t)
	if _, err := document.Resolve(otherDocument, tensors); err == nil {
		t.Fatal("drifted profile binding accepted")
	}
}

func definitionSpec() model.Spec {
	return model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: definitionArchitecture, BlockCount: definitionBlockCount, ContextLength: definitionContext,
			EmbeddingLength: definitionEmbedding, FeedForwardLength: definitionFeedForward,
			RMSNormEpsilon: definitionNormEpsilon,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: definitionHeads, HeadCountKV: definitionKVHeads,
			KeyLength: definitionHeadWidth, ValueLength: definitionHeadWidth,
			RopeFrequencyBase: definitionRopeBase,
		},
	}
}

func definitionMetadata() []gguf.Metadata {
	return []gguf.Metadata{
		definitionMetadataValue("general.architecture", gguf.ValueTypeString, definitionArchitecture),
		definitionMetadataValue("llama.block_count", gguf.ValueTypeUint32, definitionBlockCount),
		definitionMetadataValue("llama.context_length", gguf.ValueTypeUint32, definitionContext),
		definitionMetadataValue("llama.embedding_length", gguf.ValueTypeUint32, definitionEmbedding),
		definitionMetadataValue("llama.feed_forward_length", gguf.ValueTypeUint32, definitionFeedForward),
		definitionMetadataValue("llama.attention.head_count", gguf.ValueTypeUint32, definitionHeads),
		definitionMetadataValue("llama.attention.head_count_kv", gguf.ValueTypeUint32, definitionKVHeads),
		definitionMetadataValue("llama.attention.key_length", gguf.ValueTypeUint32, definitionHeadWidth),
		definitionMetadataValue("llama.attention.value_length", gguf.ValueTypeUint32, definitionHeadWidth),
		definitionMetadataValue("llama.rope.freq_base", gguf.ValueTypeFloat32, definitionRopeBase),
		definitionMetadataValue("llama.attention.layer_norm_rms_epsilon", gguf.ValueTypeFloat32, definitionNormEpsilon),
	}
}

func definitionMetadataValue(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return testutil.Metadata(key, valueType, data)
}

func writeDefinitionGGUF(t *testing.T, path string) {
	t.Helper()
	testutil.WriteGGUF(t, path, definitionMetadata(), []gguf.TensorData{{
		Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, int(definitionTensorBytes))),
	}})
}
