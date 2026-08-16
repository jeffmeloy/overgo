package modelrecipe

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestResolveActiveGGUFRequiresExactActiveProgram(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)

	t.Run("missing", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		_, err := ResolveActiveGGUF(context.Background(), store, path)
		assertProgramError(t, err, "active inference recipe is absent")
	})

	t.Run("inactive", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		inventory, resolved := publishProgramFacts(t, store, path)
		definition := definitionRecipe(t, inventory, resolved)
		if _, _, err := PublishCandidate(context.Background(), store, "fixture/inactive", definition); err != nil {
			t.Fatal(err)
		}
		_, err := ResolveActiveGGUF(context.Background(), store, path)
		assertProgramError(t, err, "active inference recipe is absent")
	})

	t.Run("incompatible", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		inventory, _ := publishProgramFacts(t, store, path)
		definition, err := inferenceFixture(
			inventory.Manifest.ID, recipe.PlacementHybrid, DecodeSessionCapacity,
		)
		if err != nil {
			t.Fatal(err)
		}
		activateProgram(t, store, definition)
		_, err = ResolveActiveGGUF(context.Background(), store, path)
		assertProgramError(t, err, "no model definition")
	})

	t.Run("mismatched", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		inventory, _ := publishProgramFacts(t, store, path)
		otherPath := t.TempDir() + "/other.gguf"
		writeServingDefinitionGGUF(t, otherPath, true)
		_, other := publishProgramFacts(t, store, otherPath)
		definition, err := InferenceWithModelDefinition(
			inventory.Manifest.ID, other.Profile.ID, other.Document.ID, recipe.PlacementHybrid,
			DecodeSessionCapacity, recipe.ResidencyHybridNative,
		)
		if err != nil {
			t.Fatal(err)
		}
		activateProgram(t, store, definition)
		_, err = ResolveActiveGGUF(context.Background(), store, path)
		assertProgramError(t, err, "differs from active model definition")
	})
}

func TestResolveActiveGGUFProducesIdentityBoundProgram(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	store := openProgramStore(t)
	defer store.Close()
	inventory, resolved := publishProgramFacts(t, store, path)
	definition := definitionRecipe(t, inventory, resolved)
	activateProgram(t, store, definition)

	loaded, err := ResolveActiveGGUF(context.Background(), store, path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	identity := loaded.state.Program.Identity
	if identity.Model != inventory.Manifest.ID || identity.Profile != resolved.Profile.ID ||
		identity.Definition != resolved.Document.ID || identity.Recipe != definition.ID ||
		identity.RecipeVersion != definition.Version || identity.Placement != recipe.PlacementHybrid ||
		identity.Runtime != RuntimeInference {
		t.Fatalf("program identity = %+v", identity)
	}
	mutated := identity
	mutated.Profile = artifact.ID{}
	if loaded.state.Program.Identity.Profile != resolved.Profile.ID {
		t.Fatal("returned identity mutated sealed program")
	}
	file, _, _, _, _, _, err := loaded.Take()
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if _, _, _, _, _, _, err := loaded.Take(); err == nil {
		t.Fatal("loaded program consumed twice")
	}
}

func TestResolveCandidateGGUFRequiresExactDefinition(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	store := openProgramStore(t)
	defer store.Close()
	inventory, resolved := publishProgramFacts(t, store, path)
	definition := definitionRecipe(t, inventory, resolved)

	loaded, err := ResolveCandidateGGUF(path, definition, resolved)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if loaded.state.Program.Identity.Recipe != definition.ID ||
		loaded.state.EvidenceTier != recipe.EvidenceExperimental {
		t.Fatalf("candidate program = %+v tier=%s", loaded.state.Program.Identity, loaded.state.EvidenceTier)
	}

	otherPath := t.TempDir() + "/other.gguf"
	writeServingDefinitionGGUF(t, otherPath, true)
	otherInventory, otherResolved := publishProgramFacts(t, store, otherPath)
	otherDefinition := definitionRecipe(t, otherInventory, otherResolved)
	_, err = ResolveCandidateGGUF(path, otherDefinition, otherResolved)
	assertProgramError(t, err, "does not name the loaded inference model")
}

func openProgramStore(t *testing.T) *repodb.Store {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func publishProgramFacts(
	t *testing.T,
	store artifact.Repository,
	path string,
) (modelartifact.Inventory, ResolvedModelDefinition) {
	t.Helper()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	profile, ok := model.LookupArchitecture(definitionArchitecture)
	if !ok {
		t.Fatal("fixture profile unavailable")
	}
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := model.ReadSpecWithProfile(file, profile)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishResolvedModelDefinition(
		context.Background(), store, "fixture/program/facts/"+inventory.Manifest.ID.String(), inventory, resolved,
	); err != nil {
		t.Fatal(err)
	}
	return inventory, resolved
}

func definitionRecipe(
	t *testing.T,
	inventory modelartifact.Inventory,
	resolved ResolvedModelDefinition,
) recipe.Definition {
	t.Helper()
	definition, err := InferenceWithModelDefinition(
		inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		DecodeSessionCapacity, recipe.ResidencyHybridNative,
	)
	if err != nil {
		t.Fatal(err)
	}
	return definition
}

func activateProgram(t *testing.T, store artifact.Repository, definition recipe.Definition) {
	t.Helper()
	ctx := context.Background()
	if _, _, err := PublishCandidate(ctx, store, "fixture/program/candidate/"+definition.ID.String(), definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/program/validated/"+definition.ID.String(), definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/program/verification/"+definition.ID.String())
	if _, _, err := ActivateVerified(ctx, store, "fixture/program/active/"+definition.ID.String(), definition, verification, nil, nil); err != nil {
		t.Fatal(err)
	}
}

func assertProgramError(t *testing.T, err error, contains string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), contains) {
		t.Fatalf("error = %v, want containing %q", err, contains)
	}
}

func writeServingDefinitionGGUF(t *testing.T, path string, distinct bool) {
	t.Helper()
	const vocabulary = uint64(8)
	metadata := append(definitionMetadata(),
		definitionMetadataValue("llama.vocab_size", gguf.ValueTypeUint32, uint32(vocabulary)),
	)
	tensor := func(name string, shape ...uint64) gguf.TensorData {
		elements := uint64(1)
		for _, dimension := range shape {
			elements *= dimension
		}
		data := make([]byte, int(elements*definitionF32Bytes))
		if distinct && name == "token_embd.weight" {
			data[len(data)-1] = 1
		}
		return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(data)}
	}
	testutil.WriteGGUF(t, path, metadata, []gguf.TensorData{
		tensor("token_embd.weight", definitionTensorWidth, vocabulary),
		tensor("output_norm.weight", definitionTensorWidth),
		tensor("output.weight", definitionTensorWidth, vocabulary),
		tensor("blk.0.attn_norm.weight", definitionTensorWidth),
		tensor("blk.0.attn_q.weight", definitionTensorWidth, definitionTensorWidth),
		tensor("blk.0.attn_k.weight", definitionTensorWidth, uint64(definitionHeadWidth)),
		tensor("blk.0.attn_v.weight", definitionTensorWidth, uint64(definitionHeadWidth)),
		tensor("blk.0.attn_output.weight", definitionTensorWidth, definitionTensorWidth),
		tensor("blk.0.ffn_norm.weight", definitionTensorWidth),
		tensor("blk.0.ffn_gate.weight", definitionTensorWidth, uint64(definitionFeedForward)),
		tensor("blk.0.ffn_up.weight", definitionTensorWidth, uint64(definitionFeedForward)),
		tensor("blk.0.ffn_down.weight", uint64(definitionFeedForward), definitionTensorWidth),
	})
}
