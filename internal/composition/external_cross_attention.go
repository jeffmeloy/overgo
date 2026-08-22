package composition

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/recipe"
)

const (
	externalCrossAttentionDefinitionMediaType = "application/vnd.overgo.external-cross-attention-definition+json"
	externalCrossAttentionDefinitionSchema    = "overgo/external-cross-attention-definition/v1"
	externalCrossAttentionPlanMediaType       = "application/vnd.overgo.external-cross-attention-plan+json"
	externalCrossAttentionPlanSchema          = "overgo/external-cross-attention-plan/v1"
	externalCrossAttentionCacheDomain         = "composition.external-cross-attention"
)

// ExternalCrossAttentionDefinition binds an adapter to explicit post-layer
// target seams and a bounded external source representation.
type ExternalCrossAttentionDefinition struct {
	Version          uint16      `json:"version"`
	Target           artifact.ID `json:"target"`
	Source           artifact.ID `json:"source"`
	Adapter          artifact.ID `json:"adapter"`
	Layers           []uint32    `json:"layers"`
	SourceChannels   uint64      `json:"source_channels"`
	HeadCount        uint64      `json:"head_count"`
	SourceTokenLimit uint64      `json:"source_token_limit"`
	ID               artifact.ID `json:"-"`
}

// ExternalCrossAttentionPlan is the active recipe-owned runtime authority.
// CacheIdentity occupies an explicit external-attention domain and is never a
// target self-attention KV-cache identity.
type ExternalCrossAttentionPlan struct {
	Version           uint16      `json:"version"`
	CompositionRecipe artifact.ID `json:"composition_recipe"`
	Definition        artifact.ID `json:"definition"`
	Target            artifact.ID `json:"target"`
	Source            artifact.ID `json:"source"`
	Adapter           artifact.ID `json:"adapter"`
	Layers            []uint32    `json:"layers"`
	SourceChannels    uint64      `json:"source_channels"`
	TargetChannels    uint64      `json:"target_channels"`
	HeadCount         uint64      `json:"head_count"`
	HeadChannels      uint64      `json:"head_channels"`
	SourceTokenLimit  uint64      `json:"source_token_limit"`
	CacheIdentity     artifact.ID `json:"cache_identity"`
	ID                artifact.ID `json:"-"`
}

type externalCrossAttentionCacheAuthority struct {
	Domain         string      `json:"domain"`
	Definition     artifact.ID `json:"definition"`
	Recipe         artifact.ID `json:"recipe"`
	TargetChannels uint64      `json:"target_channels"`
	HeadChannels   uint64      `json:"head_channels"`
}

var externalCrossAttentionDefinitionCodec = artifact.JSONDocumentCodec(
	"external cross-attention definition", artifact.KindProfile,
	externalCrossAttentionDefinitionMediaType, externalCrossAttentionDefinitionSchema,
	canonicalizeExternalCrossAttentionDefinition,
	func(value ExternalCrossAttentionDefinition) artifact.ID { return value.ID },
	func(value *ExternalCrossAttentionDefinition, id artifact.ID) { value.ID = id },
	func(value ExternalCrossAttentionDefinition) ExternalCrossAttentionDefinition {
		value.Layers = slices.Clone(value.Layers)
		return value
	},
)

var externalCrossAttentionPlanCodec = artifact.JSONDocumentCodec(
	"external cross-attention plan", artifact.KindProfile,
	externalCrossAttentionPlanMediaType, externalCrossAttentionPlanSchema,
	canonicalizeExternalCrossAttentionPlan,
	func(value ExternalCrossAttentionPlan) artifact.ID { return value.ID },
	func(value *ExternalCrossAttentionPlan, id artifact.ID) { value.ID = id },
	func(value ExternalCrossAttentionPlan) ExternalCrossAttentionPlan {
		value.Layers = slices.Clone(value.Layers)
		return value
	},
)

// NewExternalCrossAttentionDefinition validates and identifies an immutable
// recipe authority.
func NewExternalCrossAttentionDefinition(value ExternalCrossAttentionDefinition) (ExternalCrossAttentionDefinition, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return externalCrossAttentionDefinitionCodec.New(value)
}

// ValidateIdentity verifies the exact definition content identity.
func (value ExternalCrossAttentionDefinition) ValidateIdentity() error {
	return externalCrossAttentionDefinitionCodec.ValidateIdentity(value)
}

// Content returns the canonical definition document.
func (value ExternalCrossAttentionDefinition) Content() (artifact.Content, error) {
	return externalCrossAttentionDefinitionCodec.Content(value)
}

// Lineage binds the definition to both models and adapter weights.
func (value ExternalCrossAttentionDefinition) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Source, value.Target, value.Adapter)
}

// LoadExternalCrossAttentionDefinition resolves exact immutable definition authority.
func LoadExternalCrossAttentionDefinition(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (ExternalCrossAttentionDefinition, error) {
	value, err := externalCrossAttentionDefinitionCodec.Require(ctx, reader, id)
	if err != nil {
		return ExternalCrossAttentionDefinition{}, err
	}
	canonical, err := NewExternalCrossAttentionDefinition(value)
	if err != nil || canonical.ID != id {
		return ExternalCrossAttentionDefinition{}, errors.Join(
			err, errors.New("composition: external cross-attention definition authority differs"),
		)
	}
	return value, nil
}

// CompileExternalCrossAttentionPlan resolves only an active composition recipe
// and validates every requested seam against the compiled target model plan.
func CompileExternalCrossAttentionPlan(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
	targetPlan model.ModelPlan,
) (ExternalCrossAttentionPlan, error) {
	active, found, err := ActiveComposition(ctx, reader, source, target, task)
	if err != nil || !found {
		return ExternalCrossAttentionPlan{}, errors.Join(err, errors.New("composition: active external cross-attention recipe is absent"))
	}
	if !active.ExternalCrossAttention.Valid() {
		return ExternalCrossAttentionPlan{}, errors.New("composition: active recipe has no external cross-attention authority")
	}
	definition, err := LoadExternalCrossAttentionDefinition(ctx, reader, active.ExternalCrossAttention)
	if err != nil {
		return ExternalCrossAttentionPlan{}, err
	}
	if !targetPlan.Compiled() || definition.Source != source || definition.Target != target ||
		definition.Adapter != active.BridgeWeights {
		return ExternalCrossAttentionPlan{}, errors.New("composition: external cross-attention runtime authority differs")
	}
	targetChannels := uint64(targetPlan.Spec().EmbeddingLength)
	headChannels, divisible := checked.DivExact64(targetChannels, definition.HeadCount)
	if !checked.Nonzero(targetChannels) || !divisible {
		return ExternalCrossAttentionPlan{}, errors.New("composition: external cross-attention head geometry is invalid")
	}
	for _, layer := range definition.Layers {
		program, seamErr := targetPlan.LayerProgram(layer)
		if seamErr != nil {
			return ExternalCrossAttentionPlan{}, seamErr
		}
		if program.Layer().Cache == model.CacheCrossAttention {
			return ExternalCrossAttentionPlan{}, errors.New("composition: external seam conflicts with target-owned cross-attention cache")
		}
	}
	cacheIdentity, err := artifact.JSONID(artifact.KindProfile, externalCrossAttentionCacheAuthority{
		Domain: externalCrossAttentionCacheDomain, Definition: definition.ID, Recipe: active.ID,
		TargetChannels: targetChannels, HeadChannels: headChannels,
	})
	if err != nil {
		return ExternalCrossAttentionPlan{}, err
	}
	return externalCrossAttentionPlanCodec.New(ExternalCrossAttentionPlan{
		Version:           artifact.InitialDocumentVersion,
		CompositionRecipe: active.ID, Definition: definition.ID,
		Target: target, Source: source, Adapter: definition.Adapter,
		Layers: slices.Clone(definition.Layers), SourceChannels: definition.SourceChannels,
		TargetChannels: targetChannels, HeadCount: definition.HeadCount, HeadChannels: headChannels,
		SourceTokenLimit: definition.SourceTokenLimit, CacheIdentity: cacheIdentity,
	})
}

// ValidateIdentity verifies compiled plan content and cache-domain identity.
func (value ExternalCrossAttentionPlan) ValidateIdentity() error {
	return externalCrossAttentionPlanCodec.ValidateIdentity(value)
}

func canonicalizeExternalCrossAttentionDefinition(value *ExternalCrossAttentionDefinition) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Target.Kind() != artifact.KindModel || value.Source.Kind() != artifact.KindModel ||
		value.Target == value.Source || value.Adapter.Kind() != artifact.KindAdapter ||
		!checked.Nonzero(value.SourceChannels) || !checked.Nonzero(value.HeadCount) ||
		!checked.Nonzero(value.SourceTokenLimit) || !checked.Nonzero(len(value.Layers)) {
		return errors.New("composition: external cross-attention definition is invalid")
	}
	slices.Sort(value.Layers)
	if len(slices.Compact(slices.Clone(value.Layers))) != len(value.Layers) {
		return errors.New("composition: external cross-attention layer seam is duplicated")
	}
	return nil
}

func canonicalizeExternalCrossAttentionPlan(value *ExternalCrossAttentionPlan) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.CompositionRecipe.Kind() != artifact.KindRecipe || value.Definition.Kind() != artifact.KindProfile ||
		value.Target.Kind() != artifact.KindModel || value.Source.Kind() != artifact.KindModel ||
		value.Target == value.Source || value.Adapter.Kind() != artifact.KindAdapter ||
		value.CacheIdentity.Kind() != artifact.KindProfile || !checked.Nonzero(len(value.Layers)) ||
		!checked.Nonzero(value.SourceChannels) || !checked.Nonzero(value.TargetChannels) ||
		!checked.Nonzero(value.HeadCount) || !checked.Nonzero(value.HeadChannels) ||
		!checked.Nonzero(value.SourceTokenLimit) {
		return errors.New("composition: external cross-attention plan is invalid")
	}
	headChannels, divisible := checked.DivExact64(value.TargetChannels, value.HeadCount)
	if !divisible || !checked.Equal(headChannels, value.HeadChannels) {
		return errors.New("composition: external cross-attention plan head geometry differs")
	}
	if !slices.IsSorted(value.Layers) || len(slices.Compact(slices.Clone(value.Layers))) != len(value.Layers) {
		return errors.New("composition: external cross-attention plan layers are not canonical")
	}
	want, err := artifact.JSONID(artifact.KindProfile, externalCrossAttentionCacheAuthority{
		Domain: externalCrossAttentionCacheDomain, Definition: value.Definition, Recipe: value.CompositionRecipe,
		TargetChannels: value.TargetChannels, HeadChannels: value.HeadChannels,
	})
	if err != nil || want != value.CacheIdentity {
		return errors.Join(err, errors.New("composition: external cross-attention cache authority differs"))
	}
	return nil
}
