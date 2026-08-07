package modelrecipe

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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
	profile.DenseGraph = model.DenseGraphTalkie
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("model-definition-fixture"))
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		modelID, modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth},
			Storage: "f32", Bytes: definitionTensorBytes,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionDocument(profileDocument, tensors, definitionSpec())
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseModelDefinitionDocument(content)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := parsed.Resolve(profileDocument, tensors)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := InferenceWithModelDefinition(
		modelID, profileDocument.ID, document.ID, recipe.PlacementHost,
	)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := CompileModelDefinition(definition, resolved, model.Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != document.ID || plan.Model.Profile.DenseGraph != model.DenseGraphTalkie {
		t.Fatalf("resolved definition/plan = (%s, %+v)", parsed.ID, plan.Model.Profile)
	}
}

func TestGGUFModelDefinitionRepoDBResolution(t *testing.T) {
	path := filepath.Join(t.TempDir(), "definition.gguf")
	writeDefinitionGGUF(t, path)
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionFromGGUF(file, profileDocument, inventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	secondPath := filepath.Join(t.TempDir(), "definition.gguf")
	writeDefinitionGGUF(t, secondPath)
	secondFile, err := gguf.Open(secondPath)
	if err != nil {
		t.Fatal(err)
	}
	defer secondFile.Close()
	secondInventory, err := modelartifact.FromGGUF(secondFile)
	if err != nil {
		t.Fatal(err)
	}
	secondDocument, err := NewModelDefinitionFromGGUF(secondFile, profileDocument, secondInventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	if secondDocument.ID != document.ID {
		t.Fatalf("relocated model definitions differ: %s != %s", secondDocument.ID, document.ID)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	resolvedSource, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishResolvedModelDefinition(
		ctx, store, "fixture/definition/bundle", inventory, resolvedSource,
	); err != nil {
		t.Fatal(err)
	}
	resolved, err := ResolveModelDefinition(ctx, store, document.ID)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Document.Model != inventory.Manifest.ID || resolved.Spec.EmbeddingLength != definitionEmbedding ||
		resolved.Tensors.ID != inventory.TensorInventory.ID {
		t.Fatalf("resolved model definition = %+v", resolved)
	}
}

func TestModelDefinitionRejectsBindingDrift(t *testing.T) {
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("binding-drift-model"))
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		modelID, modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{Name: "weight", Shape: []uint64{}, Storage: "f32", Bytes: definitionF32Bytes}},
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionDocument(profileDocument, tensors, definitionSpec())
	if err != nil {
		t.Fatal(err)
	}
	otherProfile, _ := model.LookupArchitecture("qwen3")
	otherDocument, err := NewProfileDocument(otherProfile)
	if err != nil {
		t.Fatal(err)
	}
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
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: data}}
}

func writeDefinitionGGUF(t *testing.T, path string) {
	t.Helper()
	testutil.WriteGGUF(t, path, definitionMetadata(), []gguf.TensorData{{
		Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, int(definitionTensorBytes))),
	}})
}
