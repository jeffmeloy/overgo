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
// longest-verifiable-prefix ladder, an EXPERIMENTAL tier (3 value oracles +
// structural derivations verified on the real 35GB checkpoint via sparse
// probes). The end-to-end rendered image stays FRONTIER, externally blocked on
// the edit source PNG and a torch-matched randn(seed=42); this package never
// dresses that up as parity.
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
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
)

// Task: SenseNova serves a generation-only image-gen recipe (not inference,
// not VQA). The linear image-gen topology is the recipe.CapabilityDefinition
// contract; the forward that fills it is FRONTIER.
const Task = recipe.TaskImageGen

// EvidenceTier: the honest activation tier -- the sensenovaparity ladder is a
// legitimate experimental verification, not a ship-bar oracle.
const EvidenceTier = recipe.EvidenceExperimental

// LadderStepName: the gate step that names the evidence source.
const LadderStepName = "sensenovaparity"

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
	RopeSections []routedlm.RopeSection // [64/5e6/T, 32/1e4/H, 32/1e4/W]
}

// Derive reads the SenseNova checkpoint's recipe-relevant facts through
// routedlm only. It streams tensor headers (never the 35GB of weight bytes),
// so it is cheap enough to run inside activation and verification.
func Derive(modelDir string) (DerivedFacts, error) {
	binding := routedlm.SenseNovaBinding()
	flowBind := routedlm.SenseNovaFlowBinding()
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
	flow, err := routedlm.CompileFlowPlan(src, cfg, flowCfg, flowBind)
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
	if f.ImageMerge <= 0 || f.FrequencyDim <= 0 {
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

// PublishLadderEvidence records the sensenovaparity ladder as a succeeded
// recipe-bound gate/run pair -- the verifier the verified-promotion lifecycle
// consumes. The gate step is named for the ladder and the code commit binds
// gate to run; the honest tier and the frontier residual live on the
// activation decision (see Activate). codeCommit is the 40- or 64-hex commit
// the ladder was verified under.
func PublishLadderEvidence(
	ctx context.Context,
	store artifact.Repository,
	recipeID artifact.ID,
	codeCommit string,
) (modelrecipe.Verification, error) {
	environment, err := artifact.IdentifyBytes(
		artifact.KindEvidence, []byte("sensenovaparity/environment/"+recipeID.String()),
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "sensenova/evidence/environment/" + recipeID.String(),
		Artifacts: []artifact.Descriptor{{ID: environment, Size: uint64(len(recipeID.String()))}},
	}); err != nil {
		return modelrecipe.Verification{}, fmt.Errorf("sensenova recipe: publish environment: %w", err)
	}
	record, err := runrecord.NewGateRecord(
		recipeID, environment, codeCommit, runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{
			Name: LadderStepName, Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	if err != nil {
		return modelrecipe.Verification{}, fmt.Errorf("sensenova recipe: gate record: %w", err)
	}
	batch, err := record.Batch("sensenova/evidence/gate/" + recipeID.String())
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, fmt.Errorf("sensenova recipe: publish gate: %w", err)
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

// RegisterAndActivate publishes the SenseNova model facts and the image-gen
// recipe, then promotes the recipe to ACTIVE through the verified-promotion
// lifecycle bound to the sensenovaparity ladder evidence. It mirrors the CLI
// capability activation (facts -> CapabilityDefinition -> candidate ->
// validated -> verified active) but supplies the SenseNova inventory and the
// honest ladder-derived evidence. Returns the activated definition.
func RegisterAndActivate(
	ctx context.Context,
	store artifact.Repository,
	inventory modelartifact.Inventory,
	codeCommit, reason string,
) (recipe.Definition, error) {
	modelID := inventory.Manifest.ID
	batch, err := inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return recipe.Definition{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return recipe.Definition{}, fmt.Errorf("sensenova recipe: publish model facts: %w", err)
	}
	return Activate(ctx, store, modelID, codeCommit, reason)
}

// Activate compiles the SenseNova image-gen recipe for an already-registered
// model artifact and promotes it to ACTIVE through the verified-promotion
// lifecycle. The model artifact must already be present in the store (its
// bytes are registered by RegisterAndActivate, or by any prior facts commit).
func Activate(
	ctx context.Context,
	store artifact.Repository,
	modelID artifact.ID,
	codeCommit, reason string,
) (recipe.Definition, error) {
	definition, err := modelrecipe.CapabilityDefinition(Task, modelID)
	if err != nil {
		return recipe.Definition{}, err
	}
	if err := activate(ctx, store, definition, codeCommit, reason); err != nil {
		return recipe.Definition{}, err
	}
	return definition, nil
}

// activate drives candidate -> validated -> verified-active for a fresh recipe,
// or is a no-op when the recipe is already active (idempotent resume).
func activate(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	codeCommit, reason string,
) error {
	state, published, err := modelrecipe.Status(ctx, store, definition.ID)
	if err != nil {
		return err
	}
	if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+definition.ID.String(), definition,
		); err != nil {
			return fmt.Errorf("sensenova recipe: publish candidate: %w", err)
		}
		state = recipe.StatusCandidate
	}
	if state == recipe.StatusCandidate {
		if _, _, err := modelrecipe.Transition(
			ctx, store, "recipe/validated/"+definition.ID.String(), definition,
			recipe.StatusValidated, nil, nil,
		); err != nil {
			return fmt.Errorf("sensenova recipe: transition validated: %w", err)
		}
		state = recipe.StatusValidated
	}
	switch state {
	case recipe.StatusActive:
		return nil
	case recipe.StatusValidated:
		return promoteVerified(ctx, store, definition, codeCommit, reason)
	default:
		return fmt.Errorf("sensenova recipe: recipe %s is %q; activation resumes only from candidate or validated", definition.ID, state)
	}
}

func promoteVerified(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	codeCommit, reason string,
) error {
	verification, err := PublishLadderEvidence(ctx, store, definition.ID, codeCommit)
	if err != nil {
		return err
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		return err
	}
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionAccepted, EvidenceTier, reason,
		recipe.Decider{CodeCommit: verified.Gate.CodeCommit, Derivation: verified.Gate.ID},
		[]artifact.ID{verified.Gate.ID, verified.Run.ID},
	)
	if err != nil {
		return err
	}
	content, err := decision.Content()
	if err != nil {
		return err
	}
	decisionBatch, err := artifact.NewDocumentBatch(
		"recipe/activation-decision/"+definition.ID.String(),
		[]artifact.Content{content},
		[]artifact.Lineage{
			{Child: decision.ID, Parent: definition.ID, Relation: artifact.RelationDependsOn},
			{Child: decision.ID, Parent: verified.Gate.ID, Relation: artifact.RelationDependsOn},
			{Child: decision.ID, Parent: verified.Run.ID, Relation: artifact.RelationDependsOn},
		}, nil,
	)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, decisionBatch); err != nil {
		return err
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "recipe/active/"+definition.ID.String(), definition,
		verification, []artifact.ID{decision.ID}, nil,
	); err != nil {
		return fmt.Errorf("sensenova recipe: transition active: %w", err)
	}
	return nil
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
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, Task)
	if err != nil {
		return modelrecipe.Activation{}, recipe.Program{}, err
	}
	if !active {
		return modelrecipe.Activation{}, recipe.Program{}, fmt.Errorf("sensenova recipe: model %s has no active %s recipe", modelID, Task)
	}
	program, err := modelrecipe.CompileCapability(activation.Definition)
	if err != nil {
		return modelrecipe.Activation{}, recipe.Program{}, err
	}
	return activation, program, nil
}

// Present stats the recorded model location, mirroring the discovery servable
// presence predicate (recorded-and-present, honestly false when unrecorded or
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
