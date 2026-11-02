package modelrecipe

import (
	"context"
	"errors"
	"io"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// TestABrokenVerificationKeepsItsRecipe: a recipe whose evidence does not stand is still the recipe chosen for this
// model, so it runs as itself with its state shown, and a store that fails to answer refuses the load, saying nothing of
// the model's verification.
func TestABrokenVerificationKeepsItsRecipe(t *testing.T) {
	t.Parallel()
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	t.Run("a tuned recipe with no evidence behind it", func(t *testing.T) {
		t.Parallel()
		store := openProgramStore(t)
		t.Cleanup(func() { store.Close() })
		inventory, resolved := publishProgramFacts(t, store, path)
		tuned, err := InferenceWithModelDefinition(
			inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid, DecodeSessionRequest, recipe.ResidencyHybridNative,
		)
		if err != nil {
			t.Fatal(err)
		}
		if tuned.ID == definitionRecipe(t, inventory, resolved).ID {
			t.Fatal("the fixture's tuned recipe is the one its file derives")
		}
		if _, _, err := PublishCandidate(t.Context(), store, "fixture/keep/candidate", tuned); err != nil {
			t.Fatal(err)
		}
		batch, err := artifact.NewDocumentBatch("fixture/keep/alias", nil, nil, []artifact.AliasBinding{
			{Name: activeAlias(tuned.Model, tuned.Task), Target: tuned.ID},
		})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
			t.Fatal(err)
		}
		loaded, err := ResolveActiveGGUF(t.Context(), store, path)
		if err != nil {
			t.Fatal(err)
		}
		defer loaded.Close()
		identity, err := loaded.Identity()
		if err != nil {
			t.Fatal(err)
		}
		state, err := loaded.State()
		if err != nil {
			t.Fatal(err)
		}
		if identity.Recipe != tuned.ID || state.Verification != StateBroken || state.Reason == "" {
			t.Fatalf("recipe %v, state %+v, want the tuned %v, broken", identity.Recipe, state, tuned.ID)
		}
	})
	t.Run("a store that fails to answer", func(t *testing.T) {
		t.Parallel()
		store := openProgramStore(t)
		t.Cleanup(func() { store.Close() })
		inventory, resolved := publishProgramFacts(t, store, path)
		activateProgram(t, store, definitionRecipe(t, inventory, resolved))
		failure := errors.New("the store did not answer")
		for reads := range 64 {
			failing := &readsThenFails{Reader: store, left: reads, failure: failure}
			loaded, err := ResolveActiveGGUF(t.Context(), failing, path)
			if failing.left >= 0 {
				if err != nil {
					t.Fatal(err)
				}
				loaded.Close()
				return
			}
			if !errors.Is(err, failure) {
				t.Fatalf("a failure at read %d: %v", reads, err)
			}
		}
		t.Fatal("the load read more than the fixture allows")
	})
}

// readsThenFails answers its first left content reads and fails every read after.
type readsThenFails struct {
	artifact.Reader
	left    int
	failure error
}

func (r *readsThenFails) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if r.left--; r.left < 0 {
		return artifact.Descriptor{}, nil, false, r.failure
	}
	return r.Reader.OpenContent(ctx, id)
}
