package evaluation

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	moeCapacityPairMediaType = "application/vnd.overgo.moe-capacity-pair+json"
	moeCapacityPairSchema    = "overgo/moe-capacity-pair/v1"
)

type moeCapacityEndpoint struct {
	Run        artifact.ID `json:"run"`
	Policy     artifact.ID `json:"policy"`
	Evaluation artifact.ID `json:"evaluation"`
	Dropped    uint64      `json:"dropped"`
	Total      uint64      `json:"total"`
	Admitted   bool        `json:"admitted"`
}

type moeCapacityPair struct {
	ID             artifact.ID         `json:"-"`
	Version        uint16              `json:"version"`
	Model          artifact.ID         `json:"model"`
	Dataset        artifact.ID         `json:"dataset"`
	Split          artifact.ID         `json:"split"`
	Checkpoint     artifact.ID         `json:"checkpoint"`
	Seed           artifact.ID         `json:"seed"`
	Code           artifact.ID         `json:"code"`
	Environment    artifact.ID         `json:"environment"`
	Evaluator      artifact.ID         `json:"evaluator"`
	Constrained    moeCapacityEndpoint `json:"constrained"`
	Dropless       moeCapacityEndpoint `json:"dropless"`
	DropQualityGap bool                `json:"drop_quality_gap"`
}

var moeCapacityPairCodec = artifact.JSONDocumentCodec(
	"MoE capacity pair", artifact.KindEvidence, moeCapacityPairMediaType, moeCapacityPairSchema,
	canonicalizeMoECapacityPair,
	func(value moeCapacityPair) artifact.ID { return value.ID },
	func(value *moeCapacityPair, id artifact.ID) { value.ID = id }, nil,
)

var moeCapacityPairLineage = func(value moeCapacityPair) []artifact.Lineage {
	return artifact.DependencyLineage(value.ID,
		value.Model, value.Dataset, value.Split, value.Checkpoint, value.Seed, value.Code, value.Environment, value.Evaluator,
		value.Constrained.Run, value.Constrained.Policy, value.Constrained.Evaluation,
		value.Dropless.Run, value.Dropless.Policy, value.Dropless.Evaluation,
	)
}

func canonicalizeMoECapacityPair(value *moeCapacityPair) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Checkpoint.Kind() != artifact.KindCheckpoint || value.Seed.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Environment.Kind() != artifact.KindEvidence ||
		value.Evaluator.Kind() != artifact.KindProfile || !validMoECapacityEndpoint(value.Constrained) ||
		!validMoECapacityEndpoint(value.Dropless) || value.Constrained.Run == value.Dropless.Run ||
		value.Constrained.Policy == value.Dropless.Policy || value.Constrained.Evaluation == value.Dropless.Evaluation ||
		value.Constrained.Total != value.Dropless.Total || value.Constrained.Dropped == 0 || value.Dropless.Dropped != 0 ||
		value.DropQualityGap != (!value.Constrained.Admitted && value.Dropless.Admitted) {
		return errors.New("evaluation: invalid constrained and dropless MoE pair")
	}
	return nil
}

func validMoECapacityEndpoint(value moeCapacityEndpoint) bool {
	return value.Run.Kind() == artifact.KindRun && value.Policy.Kind() == artifact.KindRecipe &&
		value.Evaluation.Kind() == artifact.KindEvidence && value.Total > 0 && value.Dropped <= value.Total
}
