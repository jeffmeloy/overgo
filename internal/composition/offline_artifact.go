package composition

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
)

const (
	// OfflineArtifactPlanVersion is the immutable offline artifact-plan version.
	OfflineArtifactPlanVersion = artifact.InitialDocumentVersion
	// OfflineArtifactPlanMediaType identifies offline artifact-production plans.
	OfflineArtifactPlanMediaType = "application/vnd.overgo.offline-composition-artifact-plan+json"
	// OfflineArtifactPlanSchema identifies the offline artifact-plan wire schema.
	OfflineArtifactPlanSchema = "overgo/offline-composition-artifact-plan/v1"
)

// OfflineArtifactOperator names artifact-production operations. These are not
// representation bridge operators and are never admitted to the composition runtime.
type OfflineArtifactOperator string

const (
	// OfflineArtifactExactPassthrough republishes one exact model artifact.
	OfflineArtifactExactPassthrough OfflineArtifactOperator = "exact-passthrough"
	// OfflineArtifactTaskArithmetic produces weights from compatible model inputs.
	OfflineArtifactTaskArithmetic OfflineArtifactOperator = "task-arithmetic"
)

// OfflineArtifactInput binds a coefficient to one exact model definition.
type OfflineArtifactInput struct {
	Definition  artifact.ID `json:"definition"`
	Coefficient float64     `json:"coefficient"`
}

// OfflineArtifactModel binds every model input to its immutable definition,
// profile, and tensor inventory authorities.
type OfflineArtifactModel struct {
	Model       artifact.ID `json:"model"`
	Definition  artifact.ID `json:"definition"`
	Profile     artifact.ID `json:"profile"`
	Inventory   artifact.ID `json:"inventory"`
	Coefficient float64     `json:"coefficient"`
}

// OfflineArtifactPlan is an immutable recipe for producing a new model
// artifact. It deliberately has no representation contracts, bridge weights,
// capture boundary, injection boundary, or runtime activation alias.
type OfflineArtifactPlan struct {
	Version  uint16                  `json:"version"`
	Operator OfflineArtifactOperator `json:"operator"`
	Inputs   []OfflineArtifactModel  `json:"inputs"`
	ID       artifact.ID             `json:"-"`
}

var offlineArtifactPlanCodec = artifact.JSONDocumentCodec(
	"offline composition artifact plan", artifact.KindRecipe,
	OfflineArtifactPlanMediaType, OfflineArtifactPlanSchema,
	canonicalizeOfflineArtifactPlan,
	func(value OfflineArtifactPlan) artifact.ID { return value.ID },
	func(value *OfflineArtifactPlan, id artifact.ID) { value.ID = id },
	func(value OfflineArtifactPlan) OfflineArtifactPlan {
		value.Inputs = slices.Clone(value.Inputs)
		return value
	},
)

// CompileOfflineArtifactPlan resolves exact RepoDB authorities and refuses any
// input whose definition, inventory, or lineage is absent or incompatible.
func CompileOfflineArtifactPlan(
	ctx context.Context,
	reader artifact.Reader,
	operator OfflineArtifactOperator,
	inputs []OfflineArtifactInput,
) (OfflineArtifactPlan, error) {
	if ctx == nil || reader == nil {
		return OfflineArtifactPlan{}, errors.New("composition: offline artifact plan requires a repository")
	}
	if err := ctx.Err(); err != nil {
		return OfflineArtifactPlan{}, err
	}
	if operator == OfflineArtifactExactPassthrough && len(inputs) != tensor.SingletonExtent {
		return OfflineArtifactPlan{}, errors.New("composition: exact passthrough requires one model definition")
	}
	if operator == OfflineArtifactTaskArithmetic && len(inputs) < tensor.PairedExtent {
		return OfflineArtifactPlan{}, errors.New("composition: task arithmetic requires at least two model definitions")
	}
	if operator != OfflineArtifactExactPassthrough && operator != OfflineArtifactTaskArithmetic {
		return OfflineArtifactPlan{}, errors.New("composition: unsupported offline artifact operator")
	}

	resolved := make([]modelrecipe.ResolvedModelDefinition, len(inputs))
	models := make([]OfflineArtifactModel, len(inputs))
	seen := make(map[artifact.ID]bool, len(inputs))
	for index, input := range inputs {
		if input.Definition.Kind() != artifact.KindModelDefinition ||
			math.IsNaN(input.Coefficient) || math.IsInf(input.Coefficient, tensor.FirstOffset) ||
			input.Coefficient == float64(tensor.FirstOffset) {
			return OfflineArtifactPlan{}, fmt.Errorf("composition: offline artifact input %d is invalid", index)
		}
		if seen[input.Definition] {
			return OfflineArtifactPlan{}, errors.New("composition: offline artifact definitions must be unique")
		}
		seen[input.Definition] = true
		definition, err := modelrecipe.ResolveModelDefinition(ctx, reader, input.Definition)
		if err != nil {
			return OfflineArtifactPlan{}, fmt.Errorf("composition: resolve offline artifact input %d: %w", index, err)
		}
		if err := validateOfflineModelLineage(ctx, reader, definition); err != nil {
			return OfflineArtifactPlan{}, fmt.Errorf("composition: offline artifact input %d: %w", index, err)
		}
		resolved[index] = definition
		models[index] = OfflineArtifactModel{
			Model: definition.Document.Model, Definition: definition.Document.ID,
			Profile: definition.Profile.ID, Inventory: definition.Tensors.ID,
			Coefficient: input.Coefficient,
		}
	}
	if operator == OfflineArtifactExactPassthrough && inputs[tensor.FirstOffset].Coefficient != float64(tensor.SingletonExtent) {
		return OfflineArtifactPlan{}, errors.New("composition: exact passthrough coefficient must be one")
	}
	for index := tensor.SingletonExtent; index < len(resolved); index++ {
		if err := validateOfflineCompatibility(resolved[tensor.FirstOffset], resolved[index]); err != nil {
			return OfflineArtifactPlan{}, fmt.Errorf("composition: task arithmetic input %d: %w", index, err)
		}
	}
	return offlineArtifactPlanCodec.New(OfflineArtifactPlan{
		Version: OfflineArtifactPlanVersion, Operator: operator, Inputs: models,
	})
}

// Content returns the immutable artifact-production plan document.
func (value OfflineArtifactPlan) Content() (artifact.Content, error) {
	return offlineArtifactPlanCodec.Content(value)
}

// Lineage binds the plan to every exact model authority it consumes.
func (value OfflineArtifactPlan) Lineage() []artifact.Lineage {
	var parents []artifact.ID
	for _, input := range value.Inputs {
		parents = append(parents, input.Model, input.Definition, input.Profile, input.Inventory)
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares publication of the offline artifact-production plan.
func (value OfflineArtifactPlan) Batch(key string) (artifact.Batch, error) {
	return offlineArtifactPlanCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeOfflineArtifactPlan(value *OfflineArtifactPlan) error {
	if value == nil || value.Version != OfflineArtifactPlanVersion || len(value.Inputs) == 0 ||
		value.Operator != OfflineArtifactExactPassthrough && value.Operator != OfflineArtifactTaskArithmetic {
		return errors.New("composition: invalid offline artifact plan")
	}
	seen := make(map[artifact.ID]bool, len(value.Inputs))
	for _, input := range value.Inputs {
		if input.Model.Kind() != artifact.KindModel || input.Definition.Kind() != artifact.KindModelDefinition ||
			input.Profile.Kind() != artifact.KindProfile || input.Inventory.Kind() != artifact.KindTensorInventory ||
			seen[input.Definition] || math.IsNaN(input.Coefficient) || math.IsInf(input.Coefficient, tensor.FirstOffset) ||
			input.Coefficient == float64(tensor.FirstOffset) {
			return errors.New("composition: invalid offline artifact model authority")
		}
		seen[input.Definition] = true
	}
	if value.Operator == OfflineArtifactExactPassthrough &&
		(len(value.Inputs) != tensor.SingletonExtent ||
			value.Inputs[tensor.FirstOffset].Coefficient != float64(tensor.SingletonExtent)) {
		return errors.New("composition: invalid exact passthrough plan")
	}
	if value.Operator == OfflineArtifactTaskArithmetic && len(value.Inputs) < tensor.PairedExtent {
		return errors.New("composition: invalid task arithmetic plan")
	}
	return nil
}

func validateOfflineCompatibility(base, candidate modelrecipe.ResolvedModelDefinition) error {
	baseSpec, err := json.Marshal(base.Spec)
	if err != nil {
		return err
	}
	candidateSpec, err := json.Marshal(candidate.Spec)
	if err != nil {
		return err
	}
	if base.Document.Architecture != candidate.Document.Architecture || base.Profile.ID != candidate.Profile.ID ||
		!bytes.Equal(baseSpec, candidateSpec) {
		return errors.New("model definitions are incompatible")
	}
	if base.Tensors.Format != candidate.Tensors.Format || !reflect.DeepEqual(base.Tensors.Tensors, candidate.Tensors.Tensors) {
		return errors.New("tensor inventories are incompatible")
	}
	return nil
}

func validateOfflineModelLineage(
	ctx context.Context,
	reader artifact.Reader,
	resolved modelrecipe.ResolvedModelDefinition,
) error {
	if descriptor, found, err := reader.Artifact(ctx, resolved.Document.Model); err != nil {
		return err
	} else if !found || descriptor.ID != resolved.Document.Model {
		return errors.New("model artifact is absent")
	}
	required := []artifact.Lineage{
		{Child: resolved.Tensors.ID, Parent: resolved.Document.Model, Relation: artifact.RelationDerivedFrom},
		{Child: resolved.Document.ID, Parent: resolved.Document.Model, Relation: artifact.RelationDerivedFrom},
		{Child: resolved.Document.ID, Parent: resolved.Profile.ID, Relation: artifact.RelationDependsOn},
		{Child: resolved.Document.ID, Parent: resolved.Tensors.ID, Relation: artifact.RelationDependsOn},
	}
	for _, expected := range required {
		parents, err := reader.Parents(ctx, expected.Child)
		if err != nil {
			return err
		}
		if !slices.Contains(parents, expected) {
			return fmt.Errorf("missing exact lineage %s -> %s (%s)", expected.Child, expected.Parent, expected.Relation)
		}
	}
	return nil
}
