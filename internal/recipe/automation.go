package recipe

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	// AutomationDefinitionVersion is the immutable automation envelope version.
	AutomationDefinitionVersion = artifact.InitialDocumentVersion
	// AutomationDefinitionMediaType identifies automation definition documents.
	AutomationDefinitionMediaType = "application/vnd.overgo.automation-definition+json"
	// AutomationDefinitionSchema identifies the automation definition contract.
	AutomationDefinitionSchema = "overgo.automation-definition.v1"
)

// AutomationDefinition binds trigger and delivery policy to one executable
// recipe. Datasets and capability bundles remain their existing authorities;
// the definition references them without copying their policy or payloads.
type AutomationDefinition struct {
	Version           uint16        `json:"version"`
	Name              string        `json:"name"`
	Recipe            artifact.ID   `json:"recipe"`
	TriggerPolicy     artifact.ID   `json:"trigger_policy"`
	DeliveryPolicy    artifact.ID   `json:"delivery_policy"`
	Datasets          []artifact.ID `json:"datasets,omitempty"`
	CapabilityBundles []artifact.ID `json:"capability_bundles,omitempty"`
	ID                artifact.ID   `json:"-"`
}

type automationDefinitionBody struct {
	Version           uint16        `json:"version"`
	Name              string        `json:"name"`
	Recipe            artifact.ID   `json:"recipe"`
	TriggerPolicy     artifact.ID   `json:"trigger_policy"`
	DeliveryPolicy    artifact.ID   `json:"delivery_policy"`
	Datasets          []artifact.ID `json:"datasets,omitempty"`
	CapabilityBundles []artifact.ID `json:"capability_bundles,omitempty"`
}

var automationDefinitionCodec = artifact.DocumentCodec[AutomationDefinition]{
	Name: "automation definition",
	Contract: artifact.DocumentContract{
		Kind: artifact.KindRecipe, MediaType: AutomationDefinitionMediaType, Schema: AutomationDefinitionSchema,
	},
	Decode: func(data []byte, value *AutomationDefinition) error {
		var body automationDefinitionBody
		if err := strictjson.DecodeBytes(data, &body); err != nil {
			return err
		}
		*value = AutomationDefinition{
			Version: body.Version, Name: body.Name, Recipe: body.Recipe,
			TriggerPolicy: body.TriggerPolicy, DeliveryPolicy: body.DeliveryPolicy,
			Datasets: body.Datasets, CapabilityBundles: body.CapabilityBundles,
		}
		return nil
	},
	Encode: func(value AutomationDefinition) ([]byte, error) {
		return json.Marshal(automationDefinitionBody{
			Version: value.Version, Name: value.Name, Recipe: value.Recipe,
			TriggerPolicy: value.TriggerPolicy, DeliveryPolicy: value.DeliveryPolicy,
			Datasets: value.Datasets, CapabilityBundles: value.CapabilityBundles,
		})
	},
	Canonicalize: canonicalizeAutomationDefinition,
	Clone: func(value AutomationDefinition) AutomationDefinition {
		value.Datasets = slices.Clone(value.Datasets)
		value.CapabilityBundles = slices.Clone(value.CapabilityBundles)
		return value
	},
	Identity:    func(value AutomationDefinition) artifact.ID { return value.ID },
	SetIdentity: func(value *AutomationDefinition, id artifact.ID) { value.ID = id },
}

// NewAutomationDefinition validates and identifies an immutable definition.
func NewAutomationDefinition(value AutomationDefinition) (AutomationDefinition, error) {
	value.Version, value.ID = AutomationDefinitionVersion, artifact.ID{}
	return automationDefinitionCodec.New(value)
}

// RequireAutomationDefinition loads one exact definition from a repository.
func RequireAutomationDefinition(ctx context.Context, reader artifact.Reader, id artifact.ID) (AutomationDefinition, error) {
	return automationDefinitionCodec.Require(ctx, reader, id)
}

// ValidateIdentity verifies canonical identity and contract invariants.
func (value AutomationDefinition) ValidateIdentity() error {
	return automationDefinitionCodec.ValidateIdentity(value)
}

// ArtifactContent returns canonical repository content.
func (value AutomationDefinition) ArtifactContent() (artifact.Content, error) {
	return automationDefinitionCodec.Content(value)
}

// Lineage binds the definition to every immutable authority it consumes.
func (value AutomationDefinition) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, 3+len(value.Datasets)+len(value.CapabilityBundles))
	parents = append(parents, value.Recipe, value.TriggerPolicy, value.DeliveryPolicy)
	parents = append(parents, value.Datasets...)
	parents = append(parents, value.CapabilityBundles...)
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeAutomationDefinition(value *AutomationDefinition) error {
	if value == nil || value.Version != AutomationDefinitionVersion ||
		!textcheck.LowerIdentifier(value.Name, len(value.Name)) ||
		value.Recipe.Kind() != artifact.KindRecipe ||
		value.TriggerPolicy.Kind() != artifact.KindProfile ||
		value.DeliveryPolicy.Kind() != artifact.KindProfile {
		return errors.New("recipe: invalid automation definition")
	}
	if err := canonicalizeAutomationIDs(&value.Datasets, artifact.KindDataset); err != nil {
		return err
	}
	if err := canonicalizeAutomationIDs(&value.CapabilityBundles, artifact.KindProfile); err != nil {
		return err
	}
	return nil
}

func canonicalizeAutomationIDs(values *[]artifact.ID, kind artifact.Kind) error {
	for _, id := range *values {
		if id.Kind() != kind {
			return errors.New("recipe: invalid automation dependency")
		}
	}
	sort.Slice(*values, func(i, j int) bool { return (*values)[i].String() < (*values)[j].String() })
	if compact := slices.Compact(*values); len(compact) != len(*values) {
		return errors.New("recipe: duplicate automation dependency")
	}
	*values = slices.Clone(*values)
	return nil
}
