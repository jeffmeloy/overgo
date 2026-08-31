package composition

import (
	"context"
	"errors"
	"fmt"
	"math/big"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
)

const (
	// SameBaseTaskArithmeticMediaType identifies one exact same-base linear composition.
	SameBaseTaskArithmeticMediaType = "application/vnd.overgo.same-base-task-arithmetic+json"
	// SameBaseTaskArithmeticSchema identifies the immutable composition contract.
	SameBaseTaskArithmeticSchema = "overgo/same-base-task-arithmetic/v1"
)

// SameBaseTaskDelta binds one derived model definition to an exact rational
// coefficient. Rational authority is retained even though the existing tensor
// planner consumes a deterministically derived float64 projection.
type SameBaseTaskDelta struct {
	Definition  artifact.ID `json:"definition"`
	Numerator   int64       `json:"numerator"`
	Denominator uint64      `json:"denominator"`
}

// SameBaseTaskArithmeticSpec declares the cheapest composition rung: a linear
// combination of compatible model definitions that all derive from one exact
// base. It grants no execution, publication, activation, or serving authority.
type SameBaseTaskArithmeticSpec struct {
	Version          uint16                     `json:"version"`
	BaseDefinition   artifact.ID                `json:"base_definition"`
	Deltas           []SameBaseTaskDelta        `json:"deltas"`
	OutputFormat     modelartifact.TensorFormat `json:"output_format"`
	Placement        recipe.Placement           `json:"placement"`
	MaxResidentBytes uint64                     `json:"max_resident_bytes"`
	MaxShardBytes    uint64                     `json:"max_shard_bytes"`
	MaxArtifactBytes uint64                     `json:"max_artifact_bytes"`
	Rationale        string                     `json:"rationale"`
	ReopenTrigger    string                     `json:"reopen_trigger"`
	ID               artifact.ID                `json:"-"`
}

var sameBaseTaskArithmeticCodec = artifact.JSONDocumentCodec(
	"same-base task arithmetic", artifact.KindRecipe,
	SameBaseTaskArithmeticMediaType, SameBaseTaskArithmeticSchema,
	canonicalizeSameBaseTaskArithmeticSpec,
	func(value SameBaseTaskArithmeticSpec) artifact.ID { return value.ID },
	func(value *SameBaseTaskArithmeticSpec, id artifact.ID) { value.ID = id },
	func(value SameBaseTaskArithmeticSpec) SameBaseTaskArithmeticSpec {
		value.Deltas = slices.Clone(value.Deltas)
		return value
	},
)

// newSameBaseTaskArithmeticSpec canonicalizes and identifies one bounded
// same-base composition hypothesis.
func newSameBaseTaskArithmeticSpec(value SameBaseTaskArithmeticSpec) (SameBaseTaskArithmeticSpec, error) {
	return sameBaseTaskArithmeticCodec.NewInitial(value)
}

// Content returns the immutable composition specification.
func (value SameBaseTaskArithmeticSpec) Content() (artifact.Content, error) {
	return sameBaseTaskArithmeticCodec.Content(value)
}

// Lineage binds the specification to its exact base and derived definitions.
func (value SameBaseTaskArithmeticSpec) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.BaseDefinition}
	for _, delta := range value.Deltas {
		parents = append(parents, delta.Definition)
	}
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// CandidateAdmissionAdapter returns the composition owner's compiled-Go
// admission adapter. Callers append it explicitly at the cross-package entry;
// modelrecipe does not import composition and no registry is consulted.
func CandidateAdmissionAdapter() runrecord.CandidateComponentAdmissionAdapter {
	return SameBaseCandidatePlugin{}
}

// SameBaseCandidatePlugin is the narrow first-rung realization plugin. Its
// zero value can validate and derive plans but cannot execute, publish,
// activate, resolve a transport, or mutate a runtime registry.
type SameBaseCandidatePlugin struct{}

// CandidateDomain selects the closed composition domain.
func (SameBaseCandidatePlugin) CandidateDomain() string {
	return string(modelrecipe.CandidateComposition)
}

// AdmissionAdapter returns the same stateless plugin for pre-compilation checks.
func (plugin SameBaseCandidatePlugin) AdmissionAdapter() runrecord.CandidateComponentAdmissionAdapter {
	return plugin
}

// EvaluatorPlugin returns the typed common evaluation-intent validator.
func (SameBaseCandidatePlugin) EvaluatorPlugin() modelrecipe.CandidateEvaluatorPlugin {
	return modelrecipe.CandidateEvaluationIntentValidator{}
}

// ValidateCandidateComponent refuses incompatible or ambiguously derived definitions.
func (SameBaseCandidatePlugin) ValidateCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts runrecord.CandidateAdmissionFacts,
	component runrecord.CandidateAdmissionComponent,
) error {
	spec, err := requireBoundSameBaseSpec(ctx, reader, facts.Parent, facts.Subject, component.Specification)
	if err != nil {
		return err
	}
	return validateSameBaseDefinitions(ctx, reader, spec)
}

// CompileCandidateComponent derives bounded offline plans and drop-delta ablations.
func (SameBaseCandidatePlugin) CompileCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts modelrecipe.CandidateCompileFacts,
	component modelrecipe.CandidateComponent,
) (modelrecipe.CandidateComponentCompilation, error) {
	spec, err := requireBoundSameBaseSpec(ctx, reader, facts.Parent, facts.Subject, component.Specification)
	if err != nil {
		return modelrecipe.CandidateComponentCompilation{}, err
	}
	if err := validateSameBaseDefinitions(ctx, reader, spec); err != nil {
		return modelrecipe.CandidateComponentCompilation{}, err
	}
	policy, err := NewOfflineTensorResourcePolicy(
		spec.Placement, spec.MaxResidentBytes, spec.MaxShardBytes,
		spec.Rationale, spec.ReopenTrigger,
	)
	if err != nil {
		return modelrecipe.CandidateComponentCompilation{}, err
	}
	primary, err := compileSameBaseArm(ctx, reader, spec, policy, nil)
	if err != nil {
		return modelrecipe.CandidateComponentCompilation{}, err
	}
	ablations := make([]sameBaseCompiledArm, len(spec.Deltas))
	for index := range spec.Deltas {
		ablations[index], err = compileSameBaseArm(ctx, reader, spec, policy, new(index))
		if err != nil {
			return modelrecipe.CandidateComponentCompilation{}, fmt.Errorf(
				"composition: compile drop-delta ablation %d: %w", index, err,
			)
		}
	}
	return assembleSameBaseCompilation(spec, policy, primary, ablations)
}

type sameBaseCompiledArm struct {
	artifactPlan  OfflineArtifactPlan
	execution     OfflineTensorExecutionPlan
	artifactBytes uint64
}

func compileSameBaseArm(
	ctx context.Context,
	reader artifact.Reader,
	spec SameBaseTaskArithmeticSpec,
	policy OfflineTensorResourcePolicy,
	drop *int,
) (sameBaseCompiledArm, error) {
	operator, inputs, err := sameBaseInputs(spec, drop)
	if err != nil {
		return sameBaseCompiledArm{}, err
	}
	plan, err := CompileOfflineArtifactPlan(ctx, reader, operator, inputs)
	if err != nil {
		return sameBaseCompiledArm{}, err
	}
	execution, err := CompileOfflineTensorExecutionPlan(ctx, reader, plan, policy)
	if err != nil {
		return sameBaseCompiledArm{}, err
	}
	var outputBytes uint64
	for _, shard := range execution.Shards {
		var ok bool
		outputBytes, ok = checked.Add64(outputBytes, shard.Bytes)
		if !ok {
			return sameBaseCompiledArm{}, errors.New("composition: compiled artifact size overflows")
		}
	}
	if !checked.Nonzero(outputBytes) {
		return sameBaseCompiledArm{}, errors.New("composition: compiled artifact is empty")
	}
	return sameBaseCompiledArm{artifactPlan: plan, execution: execution, artifactBytes: outputBytes}, nil
}

func assembleSameBaseCompilation(
	spec SameBaseTaskArithmeticSpec,
	policy OfflineTensorResourcePolicy,
	primary sameBaseCompiledArm,
	ablations []sameBaseCompiledArm,
) (modelrecipe.CandidateComponentCompilation, error) {
	policyContent, err := policy.Content()
	if err != nil {
		return modelrecipe.CandidateComponentCompilation{}, err
	}
	contents := []artifact.Content{policyContent}
	lineage := make([]artifact.Lineage, tensor.FirstOffset)
	documents := []artifact.ID{policy.ID}
	ablationPlans := make([]modelrecipe.CandidateDomainAblation, len(ablations))
	inputs := []artifact.ID{spec.BaseDefinition}
	var peak, total uint64
	arms := append([]sameBaseCompiledArm{primary}, ablations...)
	for index, arm := range arms {
		planContent, contentErr := arm.artifactPlan.Content()
		if contentErr != nil {
			return modelrecipe.CandidateComponentCompilation{}, contentErr
		}
		executionContent, contentErr := arm.execution.Content()
		if contentErr != nil {
			return modelrecipe.CandidateComponentCompilation{}, contentErr
		}
		contents = append(contents, planContent, executionContent)
		lineage = append(lineage, arm.artifactPlan.Lineage()...)
		lineage = append(lineage, arm.execution.Lineage()...)
		documents = append(documents, arm.artifactPlan.ID, arm.execution.ID)
		peak = max(peak, arm.execution.PeakResidentBytes)
		var ok bool
		total, ok = checked.Add64(total, arm.artifactBytes)
		if !ok {
			return modelrecipe.CandidateComponentCompilation{}, errors.New("composition: total artifact budget overflows")
		}
		for _, input := range arm.artifactPlan.Inputs {
			inputs = append(inputs, input.Model, input.Definition, input.Profile, input.Inventory)
		}
		if index > tensor.FirstOffset {
			ablationIndex := index - tensor.SingletonExtent
			ablationPlans[ablationIndex] = modelrecipe.CandidateDomainAblation{
				Omitted: spec.Deltas[ablationIndex].Definition, Realization: arm.execution.ID,
			}
		}
	}
	if total > spec.MaxArtifactBytes {
		return modelrecipe.CandidateComponentCompilation{}, fmt.Errorf(
			"composition: compiled artifacts require %d bytes, exceeding budget %d", total, spec.MaxArtifactBytes,
		)
	}
	return modelrecipe.CandidateComponentCompilation{
		Plan: modelrecipe.CandidateComponentPlan{
			Domain: modelrecipe.CandidateComposition, Specification: spec.ID,
			Subject: spec.ID, Realization: primary.execution.ID,
			ResourcePolicy: policy.ID, Inputs: uniqueIDs(inputs),
			Documents:         uniqueIDs(documents),
			PeakResidentBytes: peak, ArtifactBytes: total,
		},
		Ablations: ablationPlans, Contents: contents, Lineage: lineage,
	}, nil
}

func requireBoundSameBaseSpec(
	ctx context.Context,
	reader artifact.Reader,
	parent, subject, specification artifact.ID,
) (SameBaseTaskArithmeticSpec, error) {
	if ctx == nil || reader == nil || specification.Kind() != artifact.KindRecipe {
		return SameBaseTaskArithmeticSpec{}, errors.New("composition: same-base candidate authority is absent")
	}
	spec, err := sameBaseTaskArithmeticCodec.RequireExactLineage(
		ctx, reader, specification, SameBaseTaskArithmeticSpec.Lineage,
	)
	if err != nil {
		return SameBaseTaskArithmeticSpec{}, err
	}
	canonical, err := newSameBaseTaskArithmeticSpec(spec)
	if err != nil || canonical.ID != spec.ID || spec.BaseDefinition != parent || spec.ID != subject {
		return SameBaseTaskArithmeticSpec{}, errors.Join(
			errors.New("composition: same-base specification differs from candidate authority"), err,
		)
	}
	return spec, nil
}

func validateSameBaseDefinitions(
	ctx context.Context,
	reader artifact.Reader,
	spec SameBaseTaskArithmeticSpec,
) error {
	base, err := modelrecipe.ResolveModelDefinition(ctx, reader, spec.BaseDefinition)
	if err != nil {
		return fmt.Errorf("composition: resolve exact base definition: %w", err)
	}
	if base.Tensors.Format != spec.OutputFormat {
		return errors.New("composition: exact base tensor format differs from declared output")
	}
	compatibilityInputs := []OfflineArtifactInput{{
		Definition: spec.BaseDefinition, Coefficient: float64(tensor.SingletonExtent),
	}}
	for index, delta := range spec.Deltas {
		resolved, resolveErr := modelrecipe.ResolveModelDefinition(ctx, reader, delta.Definition)
		if resolveErr != nil {
			return fmt.Errorf("composition: resolve delta definition %d: %w", index, resolveErr)
		}
		if resolved.Tensors.Format != spec.OutputFormat {
			return fmt.Errorf("composition: delta definition %d tensor format differs", index)
		}
		if err := requireSoleTypedLineageRelation(
			ctx, reader, resolved.Document.ID, base.Document.ID,
			artifact.KindModelDefinition, artifact.RelationDerivedFrom,
		); err != nil {
			return fmt.Errorf("composition: delta definition %d does not bind the exact base: %w", index, err)
		}
		if err := requireSoleTypedLineageRelation(
			ctx, reader, resolved.Document.Model, base.Document.Model,
			artifact.KindModel, artifact.RelationDerivedFrom,
		); err != nil {
			return fmt.Errorf("composition: delta model %d does not bind the exact base: %w", index, err)
		}
		compatibilityInputs = append(compatibilityInputs, OfflineArtifactInput{
			Definition: delta.Definition, Coefficient: float64(tensor.SingletonExtent),
		})
	}
	_, err = CompileOfflineArtifactPlan(
		ctx, reader, OfflineArtifactTaskArithmetic, compatibilityInputs,
	)
	return err
}

func requireSoleTypedLineageRelation(
	ctx context.Context,
	reader artifact.Reader,
	child, parent artifact.ID,
	parentKind artifact.Kind,
	relation artifact.Relation,
) error {
	edges, err := reader.Parents(ctx, child)
	if err != nil {
		return err
	}
	matching := make([]artifact.Lineage, tensor.FirstOffset, tensor.SingletonExtent)
	for _, edge := range edges {
		if edge.Relation == relation && edge.Parent.Kind() == parentKind {
			matching = append(matching, edge)
		}
	}
	expected := artifact.Lineage{Child: child, Parent: parent, Relation: relation}
	if len(matching) != tensor.SingletonExtent || matching[tensor.FirstOffset] != expected {
		return fmt.Errorf("missing exact lineage %s -> %s (%s)", child, parent, relation)
	}
	return nil
}

func sameBaseInputs(
	spec SameBaseTaskArithmeticSpec,
	drop *int,
) (OfflineArtifactOperator, []OfflineArtifactInput, error) {
	if drop != nil {
		if dropIndex := *drop; dropIndex < tensor.FirstOffset || dropIndex >= len(spec.Deltas) {
			return "", nil, errors.New("composition: invalid same-base ablation")
		}
	}
	sum := new(big.Rat)
	for index, delta := range spec.Deltas {
		if drop == nil || index != *drop {
			sum.Add(sum, taskDeltaRat(delta))
		}
	}
	one := int64(tensor.SingletonExtent)
	base := new(big.Rat).Sub(big.NewRat(one, one), sum)
	inputs := make([]OfflineArtifactInput, tensor.FirstOffset, len(spec.Deltas)+tensor.SingletonExtent)
	if checked.Nonzero(base.Sign()) {
		coefficient, err := finiteCoefficient(base)
		if err != nil {
			return "", nil, err
		}
		inputs = append(inputs, OfflineArtifactInput{Definition: spec.BaseDefinition, Coefficient: coefficient})
	}
	for index, delta := range spec.Deltas {
		if drop != nil && index == *drop {
			continue
		}
		coefficient, err := finiteCoefficient(taskDeltaRat(delta))
		if err != nil {
			return "", nil, err
		}
		inputs = append(inputs, OfflineArtifactInput{Definition: delta.Definition, Coefficient: coefficient})
	}
	if len(inputs) == tensor.SingletonExtent &&
		inputs[tensor.FirstOffset].Coefficient == float64(tensor.SingletonExtent) {
		return OfflineArtifactExactPassthrough, inputs, nil
	}
	if len(inputs) < tensor.PairedExtent {
		return "", nil, errors.New("composition: same-base arithmetic does not produce a valid linear plan")
	}
	return OfflineArtifactTaskArithmetic, inputs, nil
}

func finiteCoefficient(value *big.Rat) (float64, error) {
	coefficient, _ := value.Float64()
	if !checked.Nonzero(coefficient) || !checked.Finite64(coefficient) {
		return 0, errors.New("composition: rational coefficient has no finite nonzero float64 projection")
	}
	return coefficient, nil
}

func taskDeltaRat(value SameBaseTaskDelta) *big.Rat {
	return new(big.Rat).SetFrac(
		big.NewInt(value.Numerator), new(big.Int).SetUint64(value.Denominator),
	)
}

func canonicalizeSameBaseTaskArithmeticSpec(value *SameBaseTaskArithmeticSpec) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.BaseDefinition.Kind() != artifact.KindModelDefinition || !checked.Nonzero(len(value.Deltas)) ||
		value.OutputFormat != modelartifact.TensorFormatSafetensors ||
		value.Placement != recipe.PlacementHost || !checked.Nonzero(value.MaxResidentBytes) ||
		!checked.Nonzero(value.MaxShardBytes) || !checked.Nonzero(value.MaxArtifactBytes) ||
		value.MaxShardBytes > value.MaxArtifactBytes ||
		strings.TrimSpace(value.Rationale) == "" || value.Rationale != strings.TrimSpace(value.Rationale) ||
		strings.TrimSpace(value.ReopenTrigger) == "" || value.ReopenTrigger != strings.TrimSpace(value.ReopenTrigger) {
		return errors.New("composition: invalid same-base task-arithmetic specification")
	}
	value.Deltas = slices.Clone(value.Deltas)
	for index := range value.Deltas {
		delta := &value.Deltas[index]
		if delta.Definition.Kind() != artifact.KindModelDefinition || delta.Definition == value.BaseDefinition ||
			!checked.Nonzero(delta.Numerator) || !checked.Nonzero(delta.Denominator) {
			return errors.New("composition: invalid same-base task delta")
		}
		normalized := taskDeltaRat(*delta)
		if !normalized.Num().IsInt64() || !normalized.Denom().IsUint64() {
			return errors.New("composition: normalized task delta is out of range")
		}
		delta.Numerator = normalized.Num().Int64()
		delta.Denominator = normalized.Denom().Uint64()
	}
	slices.SortFunc(value.Deltas, func(left, right SameBaseTaskDelta) int {
		return artifact.CompareID(left.Definition, right.Definition)
	})
	seen := make(map[artifact.ID]struct{}, len(value.Deltas))
	for _, delta := range value.Deltas {
		if _, duplicate := seen[delta.Definition]; duplicate {
			return errors.New("composition: duplicate same-base task delta")
		}
		seen[delta.Definition] = struct{}{}
	}
	return nil
}
