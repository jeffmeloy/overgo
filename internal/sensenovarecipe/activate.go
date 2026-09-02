// Package sensenovarecipe registers and activates the SenseNova-U1-8B MoT
// image-generation recipe through the shared verified-promotion lifecycle, then
// resolves it back through the SAME predicate the sibling VQA capability uses
// (modelrecipe.ActiveRecord + CompileCapability). Image-gen discovery therefore
// does not drift from recipe truth.
//
// The recipe is generation-only (task=image-gen). Its DERIVED facts come from
// routedlm (SenseNovaBinding/LoadConfig/LoadFlowConfig/CompileRopePlan/
// CompileFlowPlan) -- no new family file, no magics: every dim is tensor- or
// config-owned. Its ACTIVATION evidence is the sensenovaparity
// generation leadership gate, an EXPERIMENTAL tier. Seeded state is exact;
// neutral prefix/body, guidance, terminal, and sampler match current adaptive
// evidence and beat its reusable-body wall. Compiled request-to-PNG execution
// is gated; edit input remains open.
package sensenovarecipe

import (
	"context"
	"errors"
	"fmt"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/routedlm"
	"overgo/internal/safetensors"
)

// Task: SenseNova serves a generation-only image-gen recipe (not inference,
// not VQA). The linear image-gen topology is the recipe.CapabilityDefinition
// contract; production binding remains separate from activation truth.
const Task = recipe.TaskImageGen

// EvidenceTier: the measured activation tier -- the sensenovaparity ladder
// establishes that verification evidence stands, not a ship-bar oracle.
const EvidenceTier = recipe.EvidenceVerified

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

// Inventory builds the content-addressed model artifact facts for the SenseNova
// checkpoint (sharded safetensors + config + tokenizer) through the shared
// Hugging Face repository walker, the same path FromHFRepository serves every
// sharded model. The manifest identity is content-derived; the directory
// location backs the discovery presence stat.
func Inventory(modelDir string) (modelartifact.Inventory, error) {
	repository, err := hfrepo.Open(modelDir)
	if err != nil {
		return modelartifact.Inventory{}, fmt.Errorf("sensenova recipe: open repository: %w", err)
	}
	defer repository.Close()
	inventory, err := modelartifact.FromHFRepository(repository)
	if err != nil {
		return modelartifact.Inventory{}, fmt.Errorf("sensenova recipe: inventory: %w", err)
	}
	return inventory, nil
}

// Activate compiles the SenseNova image-gen recipe for an already-registered
// model artifact using existing recipe-bound verification.
func Activate(
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	flowProfileID artifact.ID,
	verification modelrecipe.Verification,
	reason string,
) (recipe.Definition, error) {
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleRoutedImagePrepare, modelID, flowProfileID)
	if err != nil {
		return recipe.Definition{}, err
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, EvidenceTier, reason,
	); err != nil {
		return recipe.Definition{}, err
	}
	return definition, nil
}

// Resolve reports the activated SenseNova image-gen recipe through the SAME
// predicate the sibling VQA capability serves through: modelrecipe.ActiveRecord
// (active alias + verified evidence + tier) plus CompileCapability (the
// executable image-gen topology). This is the discovery/status round-trip.
func Resolve(
	ctx context.Context,
	store artifact.Reader,
	modelID artifact.ID,
) (modelrecipe.Activation, recipe.Program, error) {
	return modelrecipe.ResolveActiveCapability(ctx, store, modelID, Task)
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
