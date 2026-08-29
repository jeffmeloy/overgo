package evaluation

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	moeRoutingBaselineMediaType = "application/vnd.overgo.moe-routing-baseline-matrix+json"
	moeRoutingBaselineSchema    = "overgo/moe-routing-baseline-matrix/v1"
)

var moeRoutingBaselineKinds = [...]moeRoutingBaselineKind{
	moeRoutingNoBalancing, moeRoutingAuxiliaryLoss, moeRoutingStaticBias, moeRoutingQuantileBias,
}

type moeRoutingBaselineKind string

const (
	moeRoutingNoBalancing   moeRoutingBaselineKind = "no-balancing"
	moeRoutingAuxiliaryLoss moeRoutingBaselineKind = "auxiliary-loss"
	moeRoutingStaticBias    moeRoutingBaselineKind = "static-bias"
	moeRoutingQuantileBias  moeRoutingBaselineKind = "quantile-bias"
)

type moeRoutingBaselineArm struct {
	Kind   moeRoutingBaselineKind `json:"kind"`
	Policy artifact.ID            `json:"policy"`
	Recipe artifact.ID            `json:"recipe"`
}

type moeRoutingPairedAuthorities struct {
	Model         artifact.ID `json:"model"`
	Dataset       artifact.ID `json:"dataset"`
	Split         artifact.ID `json:"split"`
	DataOrder     artifact.ID `json:"data_order"`
	Seed          artifact.ID `json:"seed"`
	ComputeBudget artifact.ID `json:"compute_budget"`
	Evaluator     artifact.ID `json:"evaluator"`
	Checkpoint    artifact.ID `json:"checkpoint"`
	Environment   artifact.ID `json:"environment"`
	Code          artifact.ID `json:"code"`
}

type moeRoutingBaselineMatrix struct {
	ID      artifact.ID `json:"-"`
	Version uint16      `json:"version"`
	moeRoutingPairedAuthorities
	Arms []moeRoutingBaselineArm `json:"arms"`
}

var moeRoutingBaselineCodec = artifact.JSONDocumentCodec(
	"MoE routing baseline matrix", artifact.KindRecipe, moeRoutingBaselineMediaType, moeRoutingBaselineSchema,
	canonicalizeMoERoutingBaseline,
	func(value moeRoutingBaselineMatrix) artifact.ID { return value.ID },
	func(value *moeRoutingBaselineMatrix, id artifact.ID) { value.ID = id },
	func(value moeRoutingBaselineMatrix) moeRoutingBaselineMatrix {
		value.Arms = slices.Clone(value.Arms)
		return value
	},
)

var moeRoutingBaselineLineage = func(value moeRoutingBaselineMatrix) []artifact.Lineage {
	parents := []artifact.ID{
		value.Model, value.Dataset, value.Split, value.DataOrder, value.Seed, value.ComputeBudget,
		value.Evaluator, value.Checkpoint, value.Environment, value.Code,
	}
	for _, arm := range value.Arms {
		parents = append(parents, arm.Policy, arm.Recipe)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func canonicalizeMoERoutingBaseline(value *moeRoutingBaselineMatrix) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.DataOrder.Kind() != artifact.KindEvidence || value.Seed.Kind() != artifact.KindEvidence ||
		value.ComputeBudget.Kind() != artifact.KindEvidence || value.Evaluator.Kind() != artifact.KindProfile ||
		value.Checkpoint.Kind() != artifact.KindCheckpoint || value.Environment.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || len(value.Arms) != len(moeRoutingBaselineKinds) {
		return errors.New("evaluation: invalid MoE routing paired authorities")
	}
	value.Arms = slices.Clone(value.Arms)
	slices.SortFunc(value.Arms, func(left, right moeRoutingBaselineArm) int { return cmp.Compare(left.Kind, right.Kind) })
	required := make(map[moeRoutingBaselineKind]bool, len(moeRoutingBaselineKinds))
	for _, kind := range moeRoutingBaselineKinds {
		required[kind] = false
	}
	policies, recipes := make(map[artifact.ID]bool, len(value.Arms)), make(map[artifact.ID]bool, len(value.Arms))
	for _, arm := range value.Arms {
		if _, known := required[arm.Kind]; !known || required[arm.Kind] || arm.Policy.Kind() != artifact.KindRecipe ||
			arm.Recipe.Kind() != artifact.KindRecipe || policies[arm.Policy] || recipes[arm.Recipe] {
			return errors.New("evaluation: incomplete or duplicate MoE routing baseline arm")
		}
		required[arm.Kind], policies[arm.Policy], recipes[arm.Recipe] = true, true, true
	}
	return nil
}
