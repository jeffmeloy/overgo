package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestARefusedRecipeSaysWhyItIsNotVerified: the recipe a file derives may be one the store refused; it runs, as nothing
// better stands, and the model's state carries the recorded failure.
func TestARefusedRecipeSaysWhyItIsNotVerified(t *testing.T) {
	t.Parallel()
	path := t.TempDir() + "/model.gguf"
	writeServingDefinitionGGUF(t, path, false)
	store := openProgramStore(t)
	defer store.Close()
	inventory, resolved := publishProgramFacts(t, store, path)
	definition := definitionRecipe(t, inventory, resolved)
	if _, _, err := PublishCandidate(t.Context(), store, "fixture/reason/candidate", definition); err != nil {
		t.Fatal(err)
	}
	decider := testutil.ArtifactID(t, artifact.KindEvidence, "reason derivation")
	testutil.PublishArtifact(t, store, decider)
	refusal, err := recipe.NewDecision(
		definition.ID, recipe.DecisionRefused, recipe.EvidenceVerified, "kernel parity failed",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: decider},
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindRun, "reason run")},
	)
	testcheck.NoError(t, err)
	publishDecision(t, store, "fixture/reason/refusal", refusal)
	if _, _, err := Transition(t.Context(), store, "fixture/reason/refused", definition, recipe.StatusRefused, []artifact.ID{refusal.ID}, nil); err != nil {
		t.Fatal(err)
	}
	loaded, err := ResolveActiveGGUF(t.Context(), store, path)
	testcheck.NoError(t, err)
	defer loaded.Close()
	state, err := loaded.State()
	testcheck.NoError(t, err)
	if state.Verification != StateNotVerified || !strings.Contains(state.Reason, refusal.Reason) {
		t.Fatalf("state = %+v, want not verified with its recorded failure", state)
	}
}
