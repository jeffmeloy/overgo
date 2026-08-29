package dataset

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
)

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
		if !validTransformArtifact(input) || seen[input] {
			return errors.New("dataset: invalid or duplicate transform input")
		}
		seen[input] = true
	}
	for _, output := range value.Outputs {
		if !validTransformArtifact(output) || seen[output] {
			return errors.New("dataset: invalid, duplicate, or identity transform output")
		}
		seen[output] = true
	}
	return nil
}

func validTransformArtifact(id artifact.ID) bool {
	switch id.Kind() {
	case artifact.KindDataset, artifact.KindDatasetShard, artifact.KindFile:
		return true
	default:
		return false
	}
}
