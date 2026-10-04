package sensenovarecipe

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/testutil"
)

const activationReason = "SenseNova compiled text request, retained prefix/body, flow integration, and PNG publication match a fingerprinted native two-step trajectory; reusable body beats adaptive wall; edit-input binding remains open"

// TestSenseNovaImageGenRoundTripSynthetic proves the recipe lifecycle + the
// discovery/status round-trip WITHOUT the 35GB checkpoint, so it runs in CI:
// a synthetic model artifact is activated through the verified-promotion
// lifecycle, then resolved back through the SAME predicate the sibling VQA
// capability uses (Resolve => ActiveRecord + CompileCapability).
func TestSenseNovaImageGenRoundTripSynthetic(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	modelID := testutil.ArtifactID(t, artifact.KindModel, "sensenova-u1-8b-mot-infographic-v3")
	testutil.PublishArtifact(t, store, modelID)
	flowProfile := publishFlowProfileFixture(t, ctx, store)

	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, flowProfile.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "fixture/sensenova/candidate", definition,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/sensenova/verification", definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	definition, err = Activate(ctx, store, modelID, flowProfile.ID, verification, activationReason)
	if err != nil {
		t.Fatalf("activate: %v", err)
	}
	if definition.Task != recipe.TaskImageGen {
		t.Fatalf("recipe task = %q, want %q", definition.Task, recipe.TaskImageGen)
	}
	if definition.Model != modelID {
		t.Fatalf("recipe model = %s, want %s", definition.Model, modelID)
	}

	assertRoundTrip(t, ctx, store, modelID, definition)

	// Re-activation is an idempotent resume (already active => no-op), so the
	// status round-trip is stable across repeated activations.
	if _, err := Activate(ctx, store, modelID, flowProfile.ID, verification, activationReason); err != nil {
		t.Fatalf("re-activate (idempotent resume): %v", err)
	}
	assertRoundTrip(t, ctx, store, modelID, definition)
}

func publishFlowProfileFixture(t *testing.T, ctx context.Context, store *overgodb.Store) routedlm.FlowProfile {
	t.Helper()
	directory := t.TempDir()
	const metadata = `{"architectures":["NEOChatModel"],"model_type":"neo_chat"}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(metadata), 0o600); err != nil {
		t.Fatal(err)
	}
	profile, err := routedlm.InspectFlowProfile(directory)
	if err != nil {
		t.Fatal(err)
	}
	content, err := profile.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/sensenova/flow-profile", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	return profile
}

// assertRoundTrip resolves the activated recipe through the shared predicate and
// asserts the active identity, the recorded experimental tier, and a compilable
// image-gen program.
func assertRoundTrip(
	t *testing.T,
	ctx context.Context,
	store *overgodb.Store,
	modelID artifact.ID,
	definition recipe.Definition,
) {
	t.Helper()
	activation, program, err := Resolve(ctx, store, modelID)
	if err != nil {
		t.Fatalf("resolve round-trip: %v", err)
	}
	if activation.Definition.ID != definition.ID {
		t.Fatalf("round-trip recipe = %s, want %s", activation.Definition.ID, definition.ID)
	}
	if activation.Tier != EvidenceTier {
		t.Fatalf("round-trip tier = %q, want %q", activation.Tier, EvidenceTier)
	}
	if activation.Definition.Task != recipe.TaskImageGen {
		t.Fatalf("round-trip task = %q, want %q", activation.Definition.Task, recipe.TaskImageGen)
	}
	if program.Definition().ID != definition.ID {
		t.Fatalf("compiled program recipe = %s, want %s", program.Definition().ID, definition.ID)
	}
}
