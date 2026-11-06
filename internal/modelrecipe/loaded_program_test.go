package modelrecipe

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// assertRunsBroken holds a load whose recipe cannot bind to running from its own file, its state saying why.
func assertRunsBroken(t *testing.T, store artifact.Reader, path, reason string) {
	t.Helper()
	loaded := testcheck.Must(ResolveActiveGGUF(t.Context(), store, path))(t)
	defer loaded.Close()
	if state := loaded.state.State; state.Verification != StateBroken || !strings.Contains(state.Reason, reason) {
		t.Fatalf("state = %+v, want broken over %q", state, reason)
	}
}

func TestResolveActiveGGUFRequiresExactActiveProgram(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)

	t.Run("incompatible", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		inventory, _ := publishProgramFacts(t, store, path)
		definition := testcheck.Must(inferenceFixture(
			inventory.Manifest.ID, recipe.PlacementHybrid, DecodeSessionCapacity,
		))(t)
		activateProgram(t, store, definition)
		assertRunsBroken(t, store, path, "no model definition")
	})

	t.Run("mismatched", func(t *testing.T) {
		store := openProgramStore(t)
		defer store.Close()
		inventory, _ := publishProgramFacts(t, store, path)
		otherPath := t.TempDir() + "/other.gguf"
		writeServingDefinitionGGUF(t, otherPath, true)
		_, other := publishProgramFacts(t, store, otherPath)
		definition := testcheck.Must(InferenceWithModelDefinition(
			inventory.Manifest.ID, other.Profile.ID, other.Document.ID, recipe.PlacementHybrid,
			DecodeSessionCapacity, recipe.ResidencyHybridNative,
		))(t)
		activateProgram(t, store, definition)
		assertRunsBroken(t, store, path, "differs from active model definition")
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

	loaded := testcheck.Must(ResolveActiveGGUF(t.Context(), store, path))(t)
	defer loaded.Close()
	identity := loaded.state.Program.Identity
	if identity.Model != inventory.Manifest.ID || identity.Profile != resolved.Profile.ID ||
		identity.Definition != resolved.Document.ID || identity.Recipe != definition.ID ||
		identity.RecipeVersion != definition.Version || identity.Placement != recipe.PlacementHybrid ||
		identity.Runtime != RuntimeInference {
		t.Fatalf("program identity = %+v", identity)
	}
	file, _, _, _, _, claims, err := loaded.Take()
	if err != nil || len(claims) != 0 {
		t.Fatal(claims, err)
	}
	defer file.Close()
	if _, _, _, _, _, _, err := loaded.Take(); err == nil {
		t.Fatal("loaded program consumed twice")
	}
}

// TestResolveClaimedGGUFLooksTheRecipeUpByTheClaim resolves a program from a
// remembered identity without hashing: the program carries the claim, Take
// hands it over with the claim, and a claim for other bytes finds no recipe, so it derives, the claim still there for
// its taker to prove.
func TestResolveClaimedGGUFLooksTheRecipeUpByTheClaim(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	store := openProgramStore(t)
	defer store.Close()
	inventory, resolved := publishProgramFacts(t, store, path)
	activateProgram(t, store, definitionRecipe(t, inventory, resolved))
	claimTrue := func(string) (artifact.ID, uint64, bool) {
		return inventory.Components[0].ID, inventory.Components[0].Size, true
	}

	loaded := testcheck.Must(ResolveClaimedGGUF(t.Context(), store, path, claimTrue))(t)
	defer loaded.Close()
	if loaded.state.Program.Identity.Model != inventory.Manifest.ID ||
		len(loaded.state.Claims) != 1 || loaded.state.Claims[0] != inventory.Components[0].ID {
		t.Fatalf("claimed program = %+v, claims %v", loaded.state.Program.Identity, loaded.state.Claims)
	}
	file, _, _, _, _, claims, err := loaded.Take()
	if err != nil || len(claims) != 1 || claims[0] != inventory.Components[0].ID {
		t.Fatalf("Take = claims %v, %v", claims, err)
	}
	defer file.Close()

	other := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, []byte("other bytes")))(t)
	derived := testcheck.Must(ResolveClaimedGGUF(t.Context(), store, path, func(string) (artifact.ID, uint64, bool) {
		return other, inventory.Components[0].Size, true
	}))(t)
	defer derived.Close()
	if derived.state.State.Verification != StateNotVerified || len(derived.state.Claims) != 1 || derived.state.Claims[0] != other {
		t.Fatalf("a claim no recipe names = state %+v, claims %v", derived.state.State, derived.state.Claims)
	}

	hashed := testcheck.Must(ResolveClaimedGGUF(t.Context(), store, path, func(string) (artifact.ID, uint64, bool) {
		return artifact.ID{}, 0, false
	}))(t)
	if len(hashed.state.Claims) != 0 {
		t.Fatalf("an unclaimed program carries claims %v", hashed.state.Claims)
	}
	kept, _, _, _, _, _, err := hashed.Take()
	testcheck.NoError(t, err)
	defer kept.Close()
}

func TestResolveCandidateGGUFRequiresExactDefinition(t *testing.T) {
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	store := openProgramStore(t)
	defer store.Close()
	inventory, resolved := publishProgramFacts(t, store, path)
	definition := definitionRecipe(t, inventory, resolved)

	loaded, err := ResolveCandidateGGUF(t.Context(), store, path, definition, resolved, nil)
	testcheck.NoError(t, err)
	defer loaded.Close()
	if loaded.state.Program.Identity.Recipe != definition.ID ||
		loaded.state.State.Verification != StateNotVerified || len(loaded.state.Claims) != 0 {
		t.Fatalf("candidate program = %+v state=%+v claims %v", loaded.state.Program.Identity, loaded.state.State, loaded.state.Claims)
	}
	// The candidate's own inventory claims its shard: the program carries the
	// claim for the open to prove.
	claimed := testcheck.Must(ResolveCandidateGGUF(t.Context(), store, path, definition, resolved, inventory.Claim))(t)
	defer claimed.Close()
	if len(claimed.state.Claims) != 1 || claimed.state.Claims[0] != inventory.Components[0].ID {
		t.Fatalf("claimed candidate carries claims %v", claimed.state.Claims)
	}

	otherPath := t.TempDir() + "/other.gguf"
	writeServingDefinitionGGUF(t, otherPath, true)
	otherInventory, otherResolved := publishProgramFacts(t, store, otherPath)
	otherDefinition := definitionRecipe(t, otherInventory, otherResolved)
	_, err = ResolveCandidateGGUF(t.Context(), store, path, otherDefinition, otherResolved, nil)
	assertProgramError(t, err, "does not name the loaded inference model")
	// Another candidate's inventory claims nothing for this file, which is hashed.
	_, err = ResolveCandidateGGUF(t.Context(), store, path, otherDefinition, otherResolved, otherInventory.Claim)
	assertProgramError(t, err, "does not name the loaded inference model")
}

func openProgramStore(t *testing.T) *overgodb.Store {
	t.Helper()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	return store
}

func publishProgramFacts(
	t *testing.T,
	store artifact.Repository,
	path string,
) (modelartifact.Inventory, ResolvedModelDefinition) {
	t.Helper()
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	profile, ok := model.LookupArchitecture(definitionArchitecture)
	if !ok {
		t.Fatal("fixture profile unavailable")
	}
	profileDocument := testcheck.Must(NewProfileDocument(profile))(t)
	spec := testcheck.Must(model.ReadSpecWithProfile(file, profile))(t)
	document := testcheck.Must(NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec))(t)
	resolved := testcheck.Must(document.Resolve(profileDocument, inventory.TensorInventory))(t)
	testcheck.NoError(t, PublishResolvedModelDefinition(t.Context(), store, inventory, resolved))
	return inventory, resolved
}

func definitionRecipe(
	t *testing.T,
	inventory modelartifact.Inventory,
	resolved ResolvedModelDefinition,
) recipe.Definition {
	t.Helper()
	definition := testcheck.Must(InferenceWithModelDefinition(
		inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		DecodeSessionCapacity, recipe.ResidencyHybridNative,
	))(t)
	return definition
}

func activateProgram(t *testing.T, store artifact.Repository, definition recipe.Definition) {
	t.Helper()
	ctx := t.Context()
	if _, _, err := PublishCandidate(ctx, store, "fixture/program/candidate/"+definition.ID.String(), definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/program/validated/"+definition.ID.String(), definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification := publishVerification(t, store, definition.ID, "fixture/program/verification/"+definition.ID.String())
	if _, _, err := ActivateVerified(
		ctx, store, "fixture/program/active/"+definition.ID.String(), definition, verification,
		recipe.EvidenceVerified, "loaded program fixture activation", nil, nil,
	); err != nil {
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

// openedProgramStore opens a program store the test closes at its end.
func openedProgramStore(t *testing.T) artifact.Repository {
	t.Helper()
	store := openProgramStore(t)
	t.Cleanup(func() { store.Close() })
	return store
}
