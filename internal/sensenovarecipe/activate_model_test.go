//go:build modeltest

package sensenovarecipe

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
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

	inventory, err := modelartifact.FromHFPath(modelDir)
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
	if err := modelrecipe.ActivateCapability(ctx, store, definition, verification, EvidenceTier, activationReason); err != nil {
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

// DerivedFacts: recipe-relevant model facts derived from the checkpoint via
// routedlm. Every field is tensor- or config-owned (cited in sensenovaparity's
// rope_plan_derive / flow_plan_derive stages), never invented.
type DerivedFacts struct {
	Layers       int                    // llm_config.num_hidden_layers
	Hidden       int                    // fork-branch hidden width
	VisionHidden int                    // conv-embedder hidden width
	HeadDim      int                    // q_norm+q_norm_hw section widths sum
	ImageMerge   int                    // dense_embedding kernel (tensor-owned)
	FlowDim      int                    // fm_head output = channels*(patch*merge)^2
	FrequencyDim int                    // timestep mlp.0 input width
	NormSections []int                  // checkpoint QK norm spans [64,64]
	RopeSections []routedlm.RopeSection // [64/5e6/T, 32/1e4/H, 32/1e4/W]
}

// Derive reads the SenseNova checkpoint's recipe-relevant facts through
// routedlm only. It streams tensor headers (never the 35GB of weight bytes),
// so it is cheap enough to run inside activation and verification.
func Derive(modelDir string) (DerivedFacts, error) {
	binding := routedlm.SenseNovaBinding()
	flowProfile, err := routedlm.InspectFlowProfile(modelDir)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: inspect flow profile: %w", err)
	}
	cfg, err := routedlm.LoadConfig(modelDir, binding)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: load llm config: %w", err)
	}
	flowCfg, err := routedlm.LoadFlowConfig(modelDir)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: load flow config: %w", err)
	}
	src, err := safetensors.OpenSource(modelDir)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: open checkpoint: %w", err)
	}
	defer src.Close()
	rope, err := routedlm.CompileRopePlan(src, cfg, binding)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: rope plan: %w", err)
	}
	flow, err := routedlm.CompileFlowPlan(src, cfg, flowCfg, flowProfile)
	if err != nil {
		return DerivedFacts{}, fmt.Errorf("sensenova recipe: flow plan: %w", err)
	}
	facts := DerivedFacts{
		Layers:       cfg.NumHiddenLayers,
		Hidden:       flow.Hidden,
		VisionHidden: flow.VisionHidden,
		HeadDim:      cfg.HeadDim,
		ImageMerge:   flow.ImageMerge,
		FlowDim:      flow.FlowDim,
		FrequencyDim: flow.FrequencyDim,
		NormSections: append([]int(nil), rope.NormWidths...),
		RopeSections: rope.Sections,
	}
	return facts, facts.validate()
}

func (f DerivedFacts) validate() error {
	if f.Layers <= 0 || f.Hidden <= 0 || f.HeadDim <= 0 || f.FlowDim <= 0 {
		return fmt.Errorf("sensenova recipe: degenerate derived facts %+v", f)
	}
	// flow_dim = channels*(patch*merge)^2 is enforced inside CompileFlowPlan;
	// here we only assert the pixel-space head is non-trivial.
	if f.ImageMerge <= 0 || f.FrequencyDim <= 0 || len(f.NormSections) == 0 {
		return fmt.Errorf("sensenova recipe: degenerate flow facts %+v", f)
	}
	return nil
}

// Present stats the recorded model location, mirroring the discovery servable
// presence predicate (recorded-and-present, false when unrecorded or
// missing). It returns the location backing the servable claim.
func Present(ctx context.Context, store artifact.Reader, modelID artifact.ID) (string, bool, error) {
	locations, err := store.Locations(ctx, modelID)
	if err != nil {
		return "", false, err
	}
	recorded := ""
	for _, location := range locations {
		if location.Kind != artifact.LocationFile && location.Kind != artifact.LocationDirectory {
			continue
		}
		recorded = location.Value
		if _, err := os.Stat(location.Value); err == nil {
			return location.Value, true, nil
		}
	}
	if recorded == "" {
		return "", false, errors.New("sensenova recipe: model location is unrecorded")
	}
	return recorded, false, nil
}
