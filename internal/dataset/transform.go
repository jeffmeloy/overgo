package dataset

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

// DatasetTransformSpec is the public, callback-free contract for one existing
// dataset transformation. Registration identifies compiled Go behavior;
// Operation and Parameters identify its exact recipe inputs.
type DatasetTransformSpec struct {
	Registration   artifact.ID   `json:"registration"`
	Operation      artifact.ID   `json:"operation"`
	Parameters     artifact.ID   `json:"parameters"`
	Code           artifact.ID   `json:"code"`
	Environment    artifact.ID   `json:"environment"`
	Inputs         []artifact.ID `json:"inputs"`
	Outputs        []artifact.ID `json:"outputs"`
	ReplayEvidence artifact.ID   `json:"replay_evidence"`
}

// DatasetTransform wraps the canonical dataset-owned transform document.
type DatasetTransform struct{ document transform }

const (
	transformMediaType = "application/vnd.overgo.dataset-transform+json"
	transformSchema    = "overgo/dataset-transform/v1"
)

type transformIdentityMode string

const transformContentAddressed transformIdentityMode = "content-addressed"

type transform struct {
	ID              artifact.ID           `json:"-"`
	Version         uint16                `json:"version"`
	IdentityMode    transformIdentityMode `json:"identity_mode"`
	Registration    artifact.ID           `json:"registration"`
	Operation       artifact.ID           `json:"operation"`
	Parameters      artifact.ID           `json:"parameters"`
	Code            artifact.ID           `json:"code"`
	Environment     artifact.ID           `json:"environment"`
	Inputs          []artifact.ID         `json:"inputs"`
	Outputs         []artifact.ID         `json:"outputs"`
	ReplayEvidence  artifact.ID           `json:"replay_evidence"`
	Deterministic   bool                  `json:"deterministic"`
	Callable        string                `json:"callable,omitzero"`
	AdvisoryVersion string                `json:"advisory_version,omitzero"`
}

var transformCodec = artifact.JSONDocumentCodec(
	"dataset transform", artifact.KindEvidence, transformMediaType, transformSchema,
	canonicalizeTransform,
	func(value transform) artifact.ID { return value.ID },
	func(value *transform, id artifact.ID) { value.ID = id },
	func(value transform) transform {
		value.Inputs = slices.Clone(value.Inputs)
		value.Outputs = slices.Clone(value.Outputs)
		return value
	},
)

var transformLineage = func(value transform) []artifact.Lineage {
	parents := []artifact.ID{
		value.Registration, value.Operation, value.Parameters, value.Code, value.Environment, value.ReplayEvidence,
	}
	parents = append(parents, value.Inputs...)
	parents = append(parents, value.Outputs...)
	return artifact.DependencyLineage(value.ID, parents...)
}

// NewDatasetTransform identifies one deterministic, registered Go transform.
func NewDatasetTransform(spec DatasetTransformSpec) (DatasetTransform, error) {
	document, err := transformCodec.New(transform{
		Version: artifact.InitialDocumentVersion, IdentityMode: transformContentAddressed,
		Registration: spec.Registration, Operation: spec.Operation, Parameters: spec.Parameters,
		Code: spec.Code, Environment: spec.Environment, Inputs: slices.Clone(spec.Inputs),
		Outputs: slices.Clone(spec.Outputs), ReplayEvidence: spec.ReplayEvidence, Deterministic: true,
	})
	return DatasetTransform{document: document}, err
}

// RequireDatasetTransform loads the exact transform and refuses noncanonical lineage.
func RequireDatasetTransform(ctx context.Context, reader artifact.Reader, id artifact.ID) (DatasetTransform, error) {
	document, err := transformCodec.RequireExactLineage(ctx, reader, id, transformLineage)
	if err != nil {
		return DatasetTransform{}, err
	}
	return DatasetTransform{document: document}, nil
}

// ID returns the transform content identity.
func (value DatasetTransform) ID() artifact.ID { return value.document.ID }

// Spec returns an isolated transform specification.
func (value DatasetTransform) Spec() DatasetTransformSpec {
	return DatasetTransformSpec{
		Registration: value.document.Registration, Operation: value.document.Operation,
		Parameters: value.document.Parameters, Code: value.document.Code, Environment: value.document.Environment,
		Inputs: slices.Clone(value.document.Inputs), Outputs: slices.Clone(value.document.Outputs),
		ReplayEvidence: value.document.ReplayEvidence,
	}
}

// Content returns the canonical transform document.
func (value DatasetTransform) Content() (artifact.Content, error) {
	return transformCodec.Content(value.document)
}

// Lineage binds the transform to registered code, recipes, inputs, outputs, and replay evidence.
func (value DatasetTransform) Lineage() []artifact.Lineage { return transformLineage(value.document) }

// ValidateIdentity proves the transform document did not change.
func (value DatasetTransform) ValidateIdentity() error {
	return transformCodec.ValidateIdentity(value.document)
}

func canonicalizeTransform(value *transform) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.IdentityMode != transformContentAddressed ||
		value.Registration.Kind() != artifact.KindProfile || value.Operation.Kind() != artifact.KindRecipe ||
		value.Parameters.Kind() != artifact.KindRecipe || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || value.ReplayEvidence.Kind() != artifact.KindEvidence ||
		!value.Deterministic || value.Callable != "" || value.AdvisoryVersion != "" || len(value.Inputs) == 0 || len(value.Outputs) == 0 {
		return errors.New("dataset: transform requires registered content-addressed deterministic identity")
	}
	value.Inputs, value.Outputs = slices.Clone(value.Inputs), slices.Clone(value.Outputs)
	slices.SortFunc(value.Inputs, artifact.CompareID)
	slices.SortFunc(value.Outputs, artifact.CompareID)
	seen := make(map[artifact.ID]bool, len(value.Inputs)+len(value.Outputs))
	for _, input := range value.Inputs {
		if !IsContentArtifact(input) || seen[input] {
			return errors.New("dataset: invalid or duplicate transform input")
		}
		seen[input] = true
	}
	for _, output := range value.Outputs {
		if !IsContentArtifact(output) || seen[output] {
			return errors.New("dataset: invalid, duplicate, or identity transform output")
		}
		seen[output] = true
	}
	return nil
}

// IsContentArtifact reports whether id names dataset content: a dataset, a
// shard of one, or a raw file. It is the shared predicate for the kinds a
// dataset input or a record over dataset content may reference.
func IsContentArtifact(id artifact.ID) bool {
	switch id.Kind() {
	case artifact.KindDataset, artifact.KindDatasetShard, artifact.KindFile:
		return true
	default:
		return false
	}
}
