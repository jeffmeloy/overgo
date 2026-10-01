//go:build modeltest

package sensenovarecipe

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/routedlm"
	"overgo/internal/testskip"
)

// TestSenseNovaImageGenActiveOnCheckpoint runs the full path on the REAL
// read-only checkpoint: derive facts through routedlm, build the
// content-addressed inventory, register + activate, then round-trip resolve +
// prove the model bytes are present. It resolves the checkpoint through the
// store's registrations, and skips under -short.
func TestSenseNovaImageGenActiveOnCheckpoint(t *testing.T) {
	testskip.Short(t)
	modelDir := modeltest.Directory(t, "SenseNova-U1-8B-MoT-Infographic-V3")
	ctx := t.Context()

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
	flowProfile, err := routedlm.InspectFlowProfile(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	flowContent, err := flowProfile.Content()
	if err != nil {
		t.Fatal(err)
	}

	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, flowProfile.ID)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := inventory.Batch("fixture/sensenova/checkpoint/facts")
	if err != nil {
		t.Fatal(err)
	}
	batch.Contents = append(batch.Contents, flowContent)
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
	definition, err = Activate(ctx, store, modelID, flowProfile.ID, verification, activationReason)
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
