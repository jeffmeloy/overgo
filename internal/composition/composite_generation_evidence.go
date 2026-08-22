package composition

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// CompositeGenerationEvidenceVersion is the immutable evidence document version.
	CompositeGenerationEvidenceVersion = artifact.SecondDocumentVersion
	// CompositeGenerationEvidenceMediaType identifies composite-generation evidence.
	CompositeGenerationEvidenceMediaType = "application/vnd.overgo.composite-generation-evidence+json"
	// CompositeGenerationEvidenceSchema identifies the evidence wire schema.
	CompositeGenerationEvidenceSchema = "overgo/composite-generation-evidence/v2"
)

// CompositeGenerationArm identifies one required causal comparison.
type CompositeGenerationArm string

const (
	// CompositeGenerationTargetBaseline runs the target without source or bridge input.
	CompositeGenerationTargetBaseline CompositeGenerationArm = "target_baseline"
	// CompositeGenerationComposed runs the promoted source-to-target composition.
	CompositeGenerationComposed CompositeGenerationArm = "composed"
	// CompositeGenerationSourceAblated removes the source representation.
	CompositeGenerationSourceAblated CompositeGenerationArm = "source_ablated"
	// CompositeGenerationBridgeAblated removes the trained bridge transformation.
	CompositeGenerationBridgeAblated CompositeGenerationArm = "bridge_ablated"
)

var compositeGenerationArms = [...]CompositeGenerationArm{
	CompositeGenerationTargetBaseline,
	CompositeGenerationComposed,
	CompositeGenerationSourceAblated,
	CompositeGenerationBridgeAblated,
}

// CompositeGenerationFrozenModel proves one content-addressed model descriptor
// was identical before and after every generation arm.
type CompositeGenerationFrozenModel struct {
	Before artifact.Descriptor `json:"before"`
	After  artifact.Descriptor `json:"after"`
}

// CompositeGenerationTrial binds one generated output to its exact input,
// random seed, run record, and evaluation record.
type CompositeGenerationTrial struct {
	Seed        uint64                 `json:"seed"`
	Arm         CompositeGenerationArm `json:"arm"`
	Input       artifact.ID            `json:"input"`
	Output      artifact.ID            `json:"output"`
	Run         artifact.ID            `json:"run"`
	Evaluation  artifact.ID            `json:"evaluation"`
	Observation artifact.ID            `json:"observation"`
}

// CompositeGenerationEvidence records a complete four-arm generation matrix.
// Quality admission is deliberately separate: this document proves exact
// identity, completeness, causal controls, and frozen-model state.
type CompositeGenerationEvidence struct {
	Version              uint16                         `json:"version"`
	SourceModel          CompositeGenerationFrozenModel `json:"source_model"`
	TargetModel          CompositeGenerationFrozenModel `json:"target_model"`
	Bridge               artifact.Descriptor            `json:"bridge"`
	ExecutionPlan        artifact.ID                    `json:"execution_plan"`
	TargetBaselineRecipe artifact.ID                    `json:"target_baseline_recipe"`
	CompositionRecipe    artifact.ID                    `json:"composition_recipe"`
	SourceAblatedRecipe  artifact.ID                    `json:"source_ablated_recipe"`
	BridgeAblatedRecipe  artifact.ID                    `json:"bridge_ablated_recipe"`
	Dataset              artifact.ID                    `json:"dataset"`
	HeldOutSplit         artifact.ID                    `json:"held_out_split"`
	Evaluator            artifact.ID                    `json:"evaluator"`
	Trials               []CompositeGenerationTrial     `json:"trials"`
	ID                   artifact.ID                    `json:"-"`
}

// CompositeGenerationEvidenceAuthority validates and resolves immutable
// composite-generation evidence without owning model or repository mutation.
type CompositeGenerationEvidenceAuthority struct{}

var compositeGenerationEvidenceCodec = artifact.JSONDocumentCodec(
	"composite generation evidence", artifact.KindEvidence,
	CompositeGenerationEvidenceMediaType, CompositeGenerationEvidenceSchema,
	canonicalizeCompositeGenerationEvidence,
	func(value CompositeGenerationEvidence) artifact.ID { return value.ID },
	func(value *CompositeGenerationEvidence, id artifact.ID) { value.ID = id },
	func(value CompositeGenerationEvidence) CompositeGenerationEvidence {
		value.Trials = slices.Clone(value.Trials)
		return value
	},
)

// New validates, canonicalizes, and identifies a
// complete four-arm generation matrix.
func (CompositeGenerationEvidenceAuthority) New(value CompositeGenerationEvidence) (CompositeGenerationEvidence, error) {
	value.Version, value.ID = CompositeGenerationEvidenceVersion, artifact.ID{}
	return compositeGenerationEvidenceCodec.New(value)
}

// Parse admits canonical serialized evidence.
func (CompositeGenerationEvidenceAuthority) Parse(data []byte) (CompositeGenerationEvidence, error) {
	return compositeGenerationEvidenceCodec.Parse(data)
}

// Load resolves exact immutable evidence.
func (CompositeGenerationEvidenceAuthority) Load(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CompositeGenerationEvidence, error) {
	return compositeGenerationEvidenceCodec.Require(ctx, reader, id)
}

// ValidateIdentity verifies the canonical evidence envelope and content identity.
func (value CompositeGenerationEvidence) ValidateIdentity() error {
	return compositeGenerationEvidenceCodec.ValidateIdentity(value)
}

// Content returns the exact evidence document.
func (value CompositeGenerationEvidence) Content() (artifact.Content, error) {
	return compositeGenerationEvidenceCodec.Content(value)
}

// Lineage binds the evidence to every model, recipe, dataset, input, output,
// run, and evaluation identity that supports it.
func (value CompositeGenerationEvidence) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.SourceModel.Before.ID, value.TargetModel.Before.ID, value.Bridge.ID,
		value.ExecutionPlan,
		value.TargetBaselineRecipe, value.CompositionRecipe,
		value.SourceAblatedRecipe, value.BridgeAblatedRecipe,
		value.Dataset, value.HeldOutSplit, value.Evaluator,
	}
	for _, trial := range value.Trials {
		parents = append(parents, trial.Input, trial.Output, trial.Run, trial.Evaluation, trial.Observation)
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares atomic evidence publication with its exact dependency graph.
func (value CompositeGenerationEvidence) Batch(key string) (artifact.Batch, error) {
	return compositeGenerationEvidenceCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeCompositeGenerationEvidence(value *CompositeGenerationEvidence) error {
	if value == nil || value.Version != CompositeGenerationEvidenceVersion {
		return errors.New("composition: invalid composite generation evidence version")
	}
	if err := validateCompositeGenerationFrozenModel(value.SourceModel); err != nil {
		return err
	}
	if err := validateCompositeGenerationFrozenModel(value.TargetModel); err != nil {
		return err
	}
	if value.SourceModel.Before.ID == value.TargetModel.Before.ID {
		return errors.New("composition: composite generation models are not distinct")
	}
	if value.Bridge.ID.Kind() != artifact.KindAdapter || !checked.Nonzero(value.Bridge.Size) || value.Bridge.Validate() != nil {
		return errors.New("composition: composite generation bridge descriptor is invalid")
	}
	if value.ExecutionPlan.Kind() != artifact.KindProfile {
		return errors.New("composition: composite generation execution plan is invalid")
	}
	recipes := []artifact.ID{
		value.TargetBaselineRecipe, value.CompositionRecipe,
		value.SourceAblatedRecipe, value.BridgeAblatedRecipe,
	}
	seenRecipes := make(map[artifact.ID]struct{}, len(recipes))
	for _, id := range recipes {
		if id.Kind() != artifact.KindRecipe {
			return errors.New("composition: composite generation recipe identity is invalid")
		}
		seenRecipes[id] = struct{}{}
	}
	if len(seenRecipes) != len(recipes) {
		return errors.New("composition: composite generation arm recipes are not distinct")
	}
	if value.Dataset.Kind() != artifact.KindDataset ||
		value.HeldOutSplit.Kind() != artifact.KindDatasetShard ||
		value.Evaluator.Kind() != artifact.KindEvidence {
		return errors.New("composition: composite generation evaluation authority is invalid")
	}
	if len(value.Trials) == 0 {
		return errors.New("composition: composite generation trials are absent")
	}
	sort.Slice(value.Trials, func(left, right int) bool {
		if value.Trials[left].Seed != value.Trials[right].Seed {
			return value.Trials[left].Seed < value.Trials[right].Seed
		}
		if value.Trials[left].Input != value.Trials[right].Input {
			return value.Trials[left].Input.String() < value.Trials[right].Input.String()
		}
		return compositeGenerationArmOrder(value.Trials[left].Arm) < compositeGenerationArmOrder(value.Trials[right].Arm)
	})
	type caseSeed struct {
		Seed  uint64
		Input artifact.ID
	}
	groups := make(map[caseSeed]map[CompositeGenerationArm]struct{})
	runs := make(map[artifact.ID]struct{}, len(value.Trials))
	evaluations := make(map[artifact.ID]struct{}, len(value.Trials))
	observations := make(map[artifact.ID]struct{}, len(value.Trials))
	for _, trial := range value.Trials {
		if trial.Input.Kind() != artifact.KindOutput || trial.Output.Kind() != artifact.KindOutput ||
			trial.Run.Kind() != artifact.KindRun || trial.Evaluation.Kind() != artifact.KindEvaluation ||
			trial.Observation.Kind() != artifact.KindEvidence {
			return errors.New("composition: composite generation trial reference kind mismatch")
		}
		if compositeGenerationArmOrder(trial.Arm) == len(compositeGenerationArms) {
			return errors.New("composition: composite generation trial arm is invalid")
		}
		key := caseSeed{Seed: trial.Seed, Input: trial.Input}
		if groups[key] == nil {
			groups[key] = make(map[CompositeGenerationArm]struct{}, len(compositeGenerationArms))
		}
		if _, duplicate := groups[key][trial.Arm]; duplicate {
			return errors.New("composition: composite generation trial arm repeats")
		}
		groups[key][trial.Arm] = struct{}{}
		if _, duplicate := runs[trial.Run]; duplicate {
			return errors.New("composition: composite generation run repeats")
		}
		if _, duplicate := evaluations[trial.Evaluation]; duplicate {
			return errors.New("composition: composite generation evaluation repeats")
		}
		if _, duplicate := observations[trial.Observation]; duplicate {
			return errors.New("composition: composite generation resource observation repeats")
		}
		runs[trial.Run], evaluations[trial.Evaluation], observations[trial.Observation] = struct{}{}, struct{}{}, struct{}{}
	}
	for _, arms := range groups {
		if len(arms) != len(compositeGenerationArms) {
			return errors.New("composition: composite generation case lacks a required ablation arm")
		}
	}
	return nil
}

func validateCompositeGenerationFrozenModel(value CompositeGenerationFrozenModel) error {
	if value.Before.ID.Kind() != artifact.KindModel || !checked.Nonzero(value.Before.Size) ||
		value.Before.Validate() != nil || value.After.Validate() != nil || value.Before != value.After {
		return errors.New("composition: composite generation frozen model descriptor changed")
	}
	return nil
}

func compositeGenerationArmOrder(arm CompositeGenerationArm) int {
	for index, candidate := range compositeGenerationArms {
		if arm == candidate {
			return index
		}
	}
	return len(compositeGenerationArms)
}
