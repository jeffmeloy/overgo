package sensenovarecipe

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

func TestRecognizeRequiresSenseNovaArchitecture(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "config.json")
	if recognized, err := Recognize(directory); err != nil || recognized {
		t.Fatalf("missing config recognition = %v, %v", recognized, err)
	}
	if err := os.WriteFile(path, []byte(`{"architectures":["Other"],"model_type":"other"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if recognized, err := Recognize(directory); err != nil || recognized {
		t.Fatalf("foreign recognition = %v, %v", recognized, err)
	}
	if err := os.WriteFile(path, []byte(`{"architectures":["NEOChatModel"],"model_type":"neo_chat"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if recognized, err := Recognize(directory); err != nil || !recognized {
		t.Fatalf("SenseNova recognition = %v, %v", recognized, err)
	}
}

const activationReason = "SenseNova compiled text request, retained prefix/body, flow integration, and PNG publication match a fingerprinted native two-step trajectory; reusable body beats adaptive wall; edit-input binding remains open"

// TestSenseNovaImageGenRoundTripSynthetic proves the recipe lifecycle + the
// discovery/status round-trip WITHOUT the 35GB checkpoint, so it runs in CI:
// a synthetic model artifact is activated through the verified-promotion
// lifecycle, then resolved back through the SAME predicate the sibling VQA
// capability uses (Resolve => ActiveRecord + CompileCapability).
func TestSenseNovaImageGenRoundTripSynthetic(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	modelID := testutil.ArtifactID(t, artifact.KindModel, "sensenova-u1-8b-mot-infographic-v3")
	testutil.PublishArtifact(t, store, modelID)

	definition, err := modelrecipe.RoutedImageDefinition(modelID)
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
	definition, err = Activate(ctx, store, modelID, verification, activationReason)
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
	if _, err := Activate(ctx, store, modelID, verification, activationReason); err != nil {
		t.Fatalf("re-activate (idempotent resume): %v", err)
	}
	assertRoundTrip(t, ctx, store, modelID, definition)
}

// TestSenseNovaImageGenActiveOnCheckpoint runs the full path on the REAL
// read-only checkpoint: derive facts through routedlm, build the
// content-addressed inventory, register + activate, then round-trip resolve +
// prove the model bytes are present. Gated by OVERGO_SENSENOVA_MODEL so CI and
// plan-verify honestly skip (mirrors TestActiveRecipeQwen35Open).
func TestSenseNovaImageGenActiveOnCheckpoint(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	modelDir := os.Getenv("OVERGO_SENSENOVA_MODEL")
	if modelDir == "" {
		t.Skip("OVERGO_SENSENOVA_MODEL is not set")
	}
	ctx := context.Background()

	facts, err := Derive(modelDir)
	if err != nil {
		t.Fatalf("derive facts: %v", err)
	}
	assertSenseNovaFacts(t, facts)

	inventory, err := Inventory(modelDir)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	modelID := inventory.Manifest.ID
	if modelID.Kind() != artifact.KindModel {
		t.Fatalf("inventory manifest kind = %q", modelID.Kind())
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	definition, err := modelrecipe.RoutedImageDefinition(modelID)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := inventory.Batch("fixture/sensenova/checkpoint/facts")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "fixture/sensenova/checkpoint/candidate", definition,
	); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "fixture/sensenova/checkpoint/verification", definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	definition, err = Activate(ctx, store, modelID, verification, activationReason)
	if err != nil {
		t.Fatalf("register + activate: %v", err)
	}

	assertRoundTrip(t, ctx, store, modelID, definition)

	location, present, err := Present(ctx, store, modelID)
	if err != nil {
		t.Fatalf("presence: %v", err)
	}
	if !present {
		t.Fatalf("SenseNova model bytes not present at recorded location %q", location)
	}
	t.Logf("SenseNova image-gen ACTIVE tier=%s recipe=%s model=%s present=%q layers=%d flow_dim=%d",
		EvidenceTier, definition.ID, modelID, location, facts.Layers, facts.FlowDim)
}

// assertRoundTrip resolves the activated recipe through the shared predicate and
// asserts the active identity, the honest experimental tier, and a compilable
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

// assertSenseNovaFacts pins the derived facts to the SenseNova checkpoint's
// tensor/config-owned geometry (the same values sensenovaparity's
// rope_plan_derive / flow_plan_derive stages assert against the real weights).
func assertSenseNovaFacts(t *testing.T, facts DerivedFacts) {
	t.Helper()
	if facts.Layers != 42 {
		t.Fatalf("layers = %d, want 42", facts.Layers)
	}
	if facts.HeadDim != 128 {
		t.Fatalf("head_dim = %d, want 128", facts.HeadDim)
	}
	if !slices.Equal(facts.NormSections, []int{64, 64}) {
		t.Fatalf("norm sections = %v, want [64 64]", facts.NormSections)
	}
	want := []routedlm.RopeSection{
		{Width: 64, Theta: 5e6, Axis: routedlm.AxisTime},
		{Width: 32, Theta: 1e4, Axis: routedlm.AxisHeight},
		{Width: 32, Theta: 1e4, Axis: routedlm.AxisWidth},
	}
	if len(facts.RopeSections) != len(want) {
		t.Fatalf("rope sections = %+v, want %+v", facts.RopeSections, want)
	}
	for i, section := range facts.RopeSections {
		if section != want[i] {
			t.Fatalf("rope section %d = %+v, want %+v", i, section, want[i])
		}
	}
	if facts.FlowDim <= 0 || facts.ImageMerge <= 0 || facts.FrequencyDim <= 0 {
		t.Fatalf("degenerate flow facts %+v", facts)
	}
}
