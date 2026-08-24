package composition

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// CompositeGenerationCUDAEvidenceVersion is the immutable CUDA evidence version.
	CompositeGenerationCUDAEvidenceVersion = artifact.InitialDocumentVersion
	// CompositeGenerationCUDAEvidenceMediaType identifies real-artifact CUDA comparisons.
	CompositeGenerationCUDAEvidenceMediaType = "application/vnd.overgo.composite-generation-cuda-evidence+json"
	// CompositeGenerationCUDAEvidenceSchema identifies the CUDA evidence wire schema.
	CompositeGenerationCUDAEvidenceSchema = "overgo/composite-generation-cuda-evidence/v1"
)

// CompositeGenerationCUDAEvidence binds a promoted composition and its
// model-native control to identical held-out inputs on one exact CUDA device.
type CompositeGenerationCUDAEvidence struct {
	Version              uint16        `json:"version"`
	SourceModel          artifact.ID   `json:"source_model"`
	TargetModel          artifact.ID   `json:"target_model"`
	CompositionRecipe    artifact.ID   `json:"composition_recipe"`
	TargetBaselineRecipe artifact.ID   `json:"target_baseline_recipe"`
	ExecutionPlan        artifact.ID   `json:"execution_plan"`
	Promotion            artifact.ID   `json:"promotion"`
	Bridge               artifact.ID   `json:"bridge"`
	HeldOutInputs        []artifact.ID `json:"held_out_inputs"`
	BaselineOutput       artifact.ID   `json:"baseline_output"`
	ComposedOutput       artifact.ID   `json:"composed_output"`
	BaselineRun          artifact.ID   `json:"baseline_run"`
	ComposedRun          artifact.ID   `json:"composed_run"`
	BaselineObservation  artifact.ID   `json:"baseline_observation"`
	ComposedObservation  artifact.ID   `json:"composed_observation"`
	Device               artifact.ID   `json:"device"`
	BridgeExecution      artifact.ID   `json:"bridge_execution"`
	ExactOutputParity    bool          `json:"exact_output_parity"`
	ID                   artifact.ID   `json:"-"`
}

// CompositeGenerationCUDAAuthority validates immutable CUDA evidence.
type CompositeGenerationCUDAAuthority struct{}

var compositeGenerationCUDAEvidenceCodec = artifact.JSONDocumentCodec(
	"composite generation CUDA evidence", artifact.KindEvidence,
	CompositeGenerationCUDAEvidenceMediaType, CompositeGenerationCUDAEvidenceSchema,
	canonicalizeCompositeGenerationCUDAEvidence,
	func(value CompositeGenerationCUDAEvidence) artifact.ID { return value.ID },
	func(value *CompositeGenerationCUDAEvidence, id artifact.ID) { value.ID = id },
	func(value CompositeGenerationCUDAEvidence) CompositeGenerationCUDAEvidence {
		value.HeldOutInputs = slices.Clone(value.HeldOutInputs)
		return value
	},
)

// New validates, canonicalizes, and identifies exact CUDA comparison evidence.
func (CompositeGenerationCUDAAuthority) New(
	value CompositeGenerationCUDAEvidence,
) (CompositeGenerationCUDAEvidence, error) {
	value.Version, value.ID = CompositeGenerationCUDAEvidenceVersion, artifact.ID{}
	return compositeGenerationCUDAEvidenceCodec.New(value)
}

// Parse admits canonical serialized CUDA evidence.
func (CompositeGenerationCUDAAuthority) Parse(data []byte) (CompositeGenerationCUDAEvidence, error) {
	return compositeGenerationCUDAEvidenceCodec.Parse(data)
}

// Load requires exact CUDA evidence content from OvergoDB.
func (CompositeGenerationCUDAAuthority) Load(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CompositeGenerationCUDAEvidence, error) {
	content, err := loadCompositionContent(
		ctx, reader, id, artifact.KindEvidence,
		CompositeGenerationCUDAEvidenceMediaType, CompositeGenerationCUDAEvidenceSchema,
	)
	if err != nil {
		return CompositeGenerationCUDAEvidence{}, err
	}
	value, err := compositeGenerationCUDAEvidenceCodec.Parse(content.Data)
	if err != nil || value.ID != id {
		return CompositeGenerationCUDAEvidence{}, errors.Join(err, errors.New("composition: composite generation CUDA evidence identity differs"))
	}
	return value, nil
}

// ValidateIdentity verifies the CUDA evidence envelope and content identity.
func (value CompositeGenerationCUDAEvidence) ValidateIdentity() error {
	return compositeGenerationCUDAEvidenceCodec.ValidateIdentity(value)
}

// Content returns exact CUDA evidence content.
func (value CompositeGenerationCUDAEvidence) Content() (artifact.Content, error) {
	return compositeGenerationCUDAEvidenceCodec.Content(value)
}

// Lineage binds CUDA evidence to every model, input, output, run, observation,
// promotion, device, and bridge-execution fact it compares.
func (value CompositeGenerationCUDAEvidence) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.SourceModel, value.TargetModel, value.CompositionRecipe, value.TargetBaselineRecipe,
		value.ExecutionPlan, value.Promotion, value.Bridge,
		value.BaselineOutput, value.ComposedOutput, value.BaselineRun, value.ComposedRun,
		value.BaselineObservation, value.ComposedObservation, value.Device, value.BridgeExecution,
	}
	parents = append(parents, value.HeldOutInputs...)
	return artifact.DependencyLineage(value.ID, uniqueIDs(parents)...)
}

// Batch prepares atomic publication of CUDA evidence and its dependency graph.
func (value CompositeGenerationCUDAEvidence) Batch(key string) (artifact.Batch, error) {
	return compositeGenerationCUDAEvidenceCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeCompositeGenerationCUDAEvidence(value *CompositeGenerationCUDAEvidence) error {
	if value == nil || value.Version != CompositeGenerationCUDAEvidenceVersion ||
		value.SourceModel.Kind() != artifact.KindModel || value.TargetModel.Kind() != artifact.KindModel ||
		value.SourceModel == value.TargetModel ||
		value.CompositionRecipe.Kind() != artifact.KindRecipe || value.TargetBaselineRecipe.Kind() != artifact.KindRecipe ||
		value.CompositionRecipe == value.TargetBaselineRecipe || value.ExecutionPlan.Kind() != artifact.KindProfile ||
		value.Promotion.Kind() != artifact.KindEvidence || value.Bridge.Kind() != artifact.KindAdapter ||
		value.BaselineOutput.Kind() != artifact.KindOutput || value.ComposedOutput.Kind() != artifact.KindOutput ||
		value.BaselineRun.Kind() != artifact.KindRun || value.ComposedRun.Kind() != artifact.KindRun ||
		value.BaselineRun == value.ComposedRun ||
		value.BaselineObservation.Kind() != artifact.KindEvidence || value.ComposedObservation.Kind() != artifact.KindEvidence ||
		value.BaselineObservation == value.ComposedObservation || value.Device.Kind() != artifact.KindEvidence ||
		value.BridgeExecution.Kind() != artifact.KindEvidence || !value.ExactOutputParity ||
		value.BaselineOutput != value.ComposedOutput || !checked.Nonempty(value.HeldOutInputs) {
		return errors.New("composition: invalid composite generation CUDA evidence")
	}
	for _, input := range value.HeldOutInputs {
		if input.Kind() != artifact.KindOutput {
			return errors.New("composition: composite generation CUDA input kind mismatch")
		}
	}
	slices.SortFunc(value.HeldOutInputs, func(left, right artifact.ID) int {
		return cmp.Compare(left.String(), right.String())
	})
	if len(slices.Compact(value.HeldOutInputs)) != len(value.HeldOutInputs) {
		return errors.New("composition: composite generation CUDA inputs repeat")
	}
	return nil
}
