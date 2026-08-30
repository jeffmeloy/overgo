package composition

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

const (
	// OfflineArtifactOutputMediaType identifies one exact materialized model output.
	OfflineArtifactOutputMediaType = "application/vnd.overgo.offline-artifact-output+json"
	// OfflineArtifactOutputSchema identifies the immutable output contract.
	OfflineArtifactOutputSchema = "overgo/offline-artifact-output/v1"
	// OfflineArtifactObservationMediaType identifies one measured materialization observation.
	OfflineArtifactObservationMediaType = "application/vnd.overgo.offline-artifact-observation+json"
	// OfflineArtifactObservationSchema identifies the immutable observation contract.
	OfflineArtifactObservationSchema = "overgo/offline-artifact-observation/v1"
)

// OfflineArtifactOutput binds produced bytes to the exact execution plan,
// tensor inventory, and resolved model definition that describe them.
type OfflineArtifactOutput struct {
	Version         uint16      `json:"version"`
	ExecutionPlan   artifact.ID `json:"execution_plan"`
	Model           artifact.ID `json:"model"`
	TensorInventory artifact.ID `json:"tensor_inventory"`
	ModelDefinition artifact.ID `json:"model_definition"`
	TensorBytes     uint64      `json:"tensor_bytes"`
	StoredBytes     uint64      `json:"stored_bytes"`
	ID              artifact.ID `json:"-"`
}

// OfflineArtifactObservation records the measured Go execution that produced
// one exact output. It is deliberately not promotion or serving authority.
type OfflineArtifactObservation struct {
	Version           uint16      `json:"version"`
	ExecutionPlan     artifact.ID `json:"execution_plan"`
	Output            artifact.ID `json:"output"`
	Run               artifact.ID `json:"run"`
	Model             artifact.ID `json:"model"`
	Environment       artifact.ID `json:"environment"`
	DurationNS        uint64      `json:"duration_ns"`
	PeakResidentBytes uint64      `json:"peak_resident_bytes"`
	ID                artifact.ID `json:"-"`
}

var offlineArtifactOutputCodec = artifact.JSONDocumentCodec(
	"offline artifact output", artifact.KindOutput,
	OfflineArtifactOutputMediaType, OfflineArtifactOutputSchema,
	canonicalizeOfflineArtifactOutput,
	func(value OfflineArtifactOutput) artifact.ID { return value.ID },
	func(value *OfflineArtifactOutput, id artifact.ID) { value.ID = id }, nil,
)

var offlineArtifactObservationCodec = artifact.JSONDocumentCodec(
	"offline artifact observation", artifact.KindEvidence,
	OfflineArtifactObservationMediaType, OfflineArtifactObservationSchema,
	canonicalizeOfflineArtifactObservation,
	func(value OfflineArtifactObservation) artifact.ID { return value.ID },
	func(value *OfflineArtifactObservation, id artifact.ID) { value.ID = id }, nil,
)

// NewOfflineArtifactOutput validates the produced catalog against the exact
// execution plan and derives byte totals from the inventory itself.
func NewOfflineArtifactOutput(
	plan OfflineTensorExecutionPlan,
	inventory modelartifact.Inventory,
	definition modelrecipe.ModelDefinitionDocument,
) (OfflineArtifactOutput, error) {
	if err := plan.ValidateIdentity(); err != nil {
		return OfflineArtifactOutput{}, err
	}
	if err := inventory.Manifest.Validate(); err != nil {
		return OfflineArtifactOutput{}, err
	}
	if err := inventory.TensorInventory.ValidateIdentity(); err != nil {
		return OfflineArtifactOutput{}, err
	}
	if err := definition.ValidateIdentity(); err != nil {
		return OfflineArtifactOutput{}, err
	}
	if inventory.Manifest.ID != inventory.TensorInventory.Owner ||
		definition.Model != inventory.Manifest.ID ||
		definition.TensorInventory != inventory.TensorInventory.ID ||
		len(plan.Operations) != len(inventory.TensorInventory.Tensors) {
		return OfflineArtifactOutput{}, errors.New("composition: materialized output authority differs")
	}
	var tensorBytes uint64
	for index, operation := range plan.Operations {
		fact := inventory.TensorInventory.Tensors[index]
		if fact.Name != operation.Name || fact.Storage != operation.Storage || fact.Bytes != operation.OutputBytes {
			return OfflineArtifactOutput{}, fmt.Errorf("composition: materialized tensor %q differs", operation.Name)
		}
		var ok bool
		tensorBytes, ok = checked.Add64(tensorBytes, fact.Bytes)
		if !ok {
			return OfflineArtifactOutput{}, errors.New("composition: materialized tensor extent overflows")
		}
	}
	var storedBytes uint64
	for _, descriptor := range inventory.Components {
		var ok bool
		storedBytes, ok = checked.Add64(storedBytes, descriptor.Size)
		if !ok {
			return OfflineArtifactOutput{}, errors.New("composition: materialized storage extent overflows")
		}
	}
	return offlineArtifactOutputCodec.New(OfflineArtifactOutput{
		Version: artifact.InitialDocumentVersion, ExecutionPlan: plan.ID,
		Model: inventory.Manifest.ID, TensorInventory: inventory.TensorInventory.ID,
		ModelDefinition: definition.ID, TensorBytes: tensorBytes, StoredBytes: storedBytes,
	})
}

// NewOfflineArtifactObservation binds one succeeded, environment-bound run to
// the exact materialized output and measured resource envelope.
func NewOfflineArtifactObservation(
	plan OfflineTensorExecutionPlan,
	output OfflineArtifactOutput,
	run runrecord.Run,
	durationNS, peakResidentBytes uint64,
) (OfflineArtifactObservation, error) {
	if err := plan.ValidateIdentity(); err != nil {
		return OfflineArtifactObservation{}, err
	}
	if err := output.ValidateIdentity(); err != nil {
		return OfflineArtifactObservation{}, err
	}
	if err := run.ValidateIdentity(); err != nil {
		return OfflineArtifactObservation{}, err
	}
	if output.ExecutionPlan != plan.ID || run.Outcome != runrecord.OutcomeSucceeded ||
		run.Recipe != plan.ArtifactPlan || !slices.Contains(run.Inputs, plan.ID) ||
		!slices.Contains(run.Outputs, output.ID) || slices.Contains(run.Inputs, output.ID) ||
		!runBindsMaterializationArtifact(run, output.Model) ||
		!runBindsMaterializationArtifact(run, output.TensorInventory) ||
		!runBindsMaterializationArtifact(run, output.ModelDefinition) ||
		run.MeasuredNS != durationNS || !slices.Contains(run.Phases, runrecord.PhaseMetric{
		Phase: runrecord.PhaseBuild, DurationNS: durationNS,
	}) {
		return OfflineArtifactObservation{}, errors.New("composition: materialization run differs from output")
	}
	return offlineArtifactObservationCodec.New(OfflineArtifactObservation{
		Version: artifact.InitialDocumentVersion, ExecutionPlan: plan.ID,
		Output: output.ID, Run: run.ID, Model: output.Model, Environment: run.Environment,
		DurationNS: durationNS, PeakResidentBytes: peakResidentBytes,
	})
}

// RequireOfflineArtifactOutput loads one exact materialized output. Its
// dependency lineage and the separate produced-by Run edge are closed together
// by the candidate-materialization owner, because the output cannot embed its
// producer without creating an identity cycle.
func RequireOfflineArtifactOutput(
	ctx context.Context, reader artifact.Reader, id artifact.ID,
) (OfflineArtifactOutput, error) {
	return offlineArtifactOutputCodec.Require(ctx, reader, id)
}

// RequireOfflineArtifactObservation loads one exact measured observation.
func RequireOfflineArtifactObservation(
	ctx context.Context, reader artifact.Reader, id artifact.ID,
) (OfflineArtifactObservation, error) {
	return offlineArtifactObservationCodec.RequireExactLineage(
		ctx, reader, id, OfflineArtifactObservation.Lineage,
	)
}

func runBindsMaterializationArtifact(run runrecord.Run, id artifact.ID) bool {
	input := slices.Contains(run.Inputs, id)
	output := slices.Contains(run.Outputs, id)
	return input != output
}

// ValidateIdentity verifies the exact output content identity.
func (value OfflineArtifactOutput) ValidateIdentity() error {
	return offlineArtifactOutputCodec.ValidateIdentity(value)
}

// Content returns the exact output document.
func (value OfflineArtifactOutput) Content() (artifact.Content, error) {
	return offlineArtifactOutputCodec.Content(value)
}

// Lineage binds the output to its exact plan and model authorities.
func (value OfflineArtifactOutput) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.ExecutionPlan, value.Model, value.TensorInventory, value.ModelDefinition,
	)
}

// ValidateIdentity verifies the exact observation content identity.
func (value OfflineArtifactObservation) ValidateIdentity() error {
	return offlineArtifactObservationCodec.ValidateIdentity(value)
}

// Content returns the exact observation document.
func (value OfflineArtifactObservation) Content() (artifact.Content, error) {
	return offlineArtifactObservationCodec.Content(value)
}

// Lineage binds the observation to its plan, output, run, model, and environment.
func (value OfflineArtifactObservation) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(
		value.ID, value.ExecutionPlan, value.Output, value.Run, value.Model, value.Environment,
	)
}

func canonicalizeOfflineArtifactOutput(value *OfflineArtifactOutput) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.ExecutionPlan.Kind() != artifact.KindProfile || value.Model.Kind() != artifact.KindModel ||
		value.TensorInventory.Kind() != artifact.KindTensorInventory ||
		value.ModelDefinition.Kind() != artifact.KindModelDefinition ||
		!checked.Nonzero(value.TensorBytes) || !checked.Nonzero(value.StoredBytes) {
		return errors.New("composition: invalid offline artifact output")
	}
	return nil
}

func canonicalizeOfflineArtifactObservation(value *OfflineArtifactObservation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.ExecutionPlan.Kind() != artifact.KindProfile || value.Output.Kind() != artifact.KindOutput ||
		value.Run.Kind() != artifact.KindRun || value.Model.Kind() != artifact.KindModel ||
		value.Environment.Kind() != artifact.KindEvidence ||
		!checked.Nonzero(value.DurationNS) || !checked.Nonzero(value.PeakResidentBytes) {
		return errors.New("composition: invalid offline artifact observation")
	}
	return nil
}
