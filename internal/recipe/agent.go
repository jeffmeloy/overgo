// Package recipe defines immutable executable and orchestration intent.
package recipe

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// AgentDefinitionMediaType identifies immutable agent definitions.
	AgentDefinitionMediaType = "application/vnd.overgo.agent-definition+json"
	// AgentDefinitionSchema identifies the agent definition contract.
	AgentDefinitionSchema = "overgo/agent-definition/v1"
)

// AgentDefinition binds one prompt and model recipe to existing capability authorities.
type AgentDefinition struct {
	Version           uint16        `json:"version"`
	Name              string        `json:"name"`
	Prompt            artifact.ID   `json:"prompt"`
	ModelRecipe       artifact.ID   `json:"model_recipe"`
	ToolManuals       []artifact.ID `json:"tool_manuals,omitempty"`
	CapabilityBundles []artifact.ID `json:"capability_bundles,omitempty"`
	Datasets          []artifact.ID `json:"datasets,omitempty"`
	Automations       []artifact.ID `json:"automations,omitempty"`
	Policies          []artifact.ID `json:"policies"`
	ID                artifact.ID   `json:"-"`
}

var agentDefinitionCodec = artifact.JSONDocumentCodec(
	"agent definition", artifact.KindRecipe, AgentDefinitionMediaType, AgentDefinitionSchema,
	canonicalizeAgentDefinition,
	func(value AgentDefinition) artifact.ID { return value.ID },
	func(value *AgentDefinition, id artifact.ID) { value.ID = id },
	func(value AgentDefinition) AgentDefinition {
		value.ToolManuals = slices.Clone(value.ToolManuals)
		value.CapabilityBundles = slices.Clone(value.CapabilityBundles)
		value.Datasets = slices.Clone(value.Datasets)
		value.Automations = slices.Clone(value.Automations)
		value.Policies = slices.Clone(value.Policies)
		return value
	},
)

// NewAgentDefinition validates and identifies one immutable agent definition.
func NewAgentDefinition(value AgentDefinition) (AgentDefinition, error) {
	return agentDefinitionCodec.NewInitial(value)
}

// RequireAgentDefinition loads one exact agent definition.
func RequireAgentDefinition(ctx context.Context, reader artifact.Reader, id artifact.ID) (AgentDefinition, error) {
	return agentDefinitionCodec.Require(ctx, reader, id)
}

// ValidateIdentity proves the definition still matches its immutable identity.
func (value AgentDefinition) ValidateIdentity() error {
	return agentDefinitionCodec.ValidateIdentity(value)
}

// ArtifactContent returns canonical repository bytes.
func (value AgentDefinition) ArtifactContent() (artifact.Content, error) {
	return agentDefinitionCodec.Content(value)
}

// Lineage binds every authority consumed by the agent definition.
func (value AgentDefinition) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Prompt, value.ModelRecipe}
	parents = append(parents, value.ToolManuals...)
	parents = append(parents, value.CapabilityBundles...)
	parents = append(parents, value.Datasets...)
	parents = append(parents, value.Automations...)
	parents = append(parents, value.Policies...)
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeAgentDefinition(value *AgentDefinition) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!textcheck.LowerIdentifier(value.Name, len(value.Name)) || value.Prompt.Kind() != artifact.KindFile ||
		value.ModelRecipe.Kind() != artifact.KindRecipe || len(value.Policies) == 0 {
		return errors.New("recipe: invalid agent definition")
	}
	groups := []struct {
		values *[]artifact.ID
		kind   artifact.Kind
	}{
		{&value.ToolManuals, artifact.KindRecipe},
		{&value.CapabilityBundles, artifact.KindProfile},
		{&value.Datasets, artifact.KindDataset},
		{&value.Automations, artifact.KindRecipe},
		{&value.Policies, artifact.KindProfile},
	}
	for _, group := range groups {
		for _, id := range *group.values {
			if id.Kind() != group.kind {
				return errors.New("recipe: invalid agent capability identity")
			}
		}
		sort.Slice(*group.values, func(left, right int) bool {
			return artifact.CompareID((*group.values)[left], (*group.values)[right]) < 0
		})
		if compact := slices.Compact(*group.values); len(compact) != len(*group.values) {
			return errors.New("recipe: duplicate agent capability identity")
		}
		*group.values = slices.Clone(*group.values)
	}
	return nil
}
