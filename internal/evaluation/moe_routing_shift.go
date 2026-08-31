package evaluation

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	moeRoutingShiftMediaType = "application/vnd.overgo.moe-routing-shift-evidence+json"
	moeRoutingShiftSchema    = "overgo/moe-routing-shift-evidence/v1"
)

type moeRoutingShiftKind string

const (
	moeRoutingHomogeneous moeRoutingShiftKind = "homogeneous"
	moeRoutingMixed       moeRoutingShiftKind = "mixed"
	moeRoutingImbalanced  moeRoutingShiftKind = "imbalanced"
	moeRoutingTransition  moeRoutingShiftKind = "transition-window"
)

var moeRoutingShiftKinds = [...]moeRoutingShiftKind{
	moeRoutingHomogeneous, moeRoutingMixed, moeRoutingImbalanced, moeRoutingTransition,
}

type moeRoutingShiftSegment struct {
	Stratum    artifact.ID  `json:"stratum"`
	Evaluation artifact.ID  `json:"evaluation"`
	Admitted   bool         `json:"admitted"`
	BiasSource *artifact.ID `json:"bias_source,omitempty"`
}

type moeRoutingShiftCase struct {
	Kind                moeRoutingShiftKind      `json:"kind"`
	AggregateEvaluation artifact.ID              `json:"aggregate_evaluation"`
	AggregateAdmitted   bool                     `json:"aggregate_admitted"`
	Segments            []moeRoutingShiftSegment `json:"segments"`
}

type moeRoutingShiftEvidence struct {
	ID                   artifact.ID           `json:"-"`
	Version              uint16                `json:"version"`
	BaselineMatrix       artifact.ID           `json:"baseline_matrix"`
	Cases                []moeRoutingShiftCase `json:"cases"`
	HiddenStratumFailure bool                  `json:"hidden_stratum_failure"`
	ControllerLag        bool                  `json:"controller_lag"`
}

var moeRoutingShiftCodec = artifact.JSONDocumentCodec(
	"MoE routing shift evidence", artifact.KindEvidence, moeRoutingShiftMediaType, moeRoutingShiftSchema,
	canonicalizeMoERoutingShift,
	func(value moeRoutingShiftEvidence) artifact.ID { return value.ID },
	func(value *moeRoutingShiftEvidence, id artifact.ID) { value.ID = id },
	func(value moeRoutingShiftEvidence) moeRoutingShiftEvidence {
		value.Cases = cloneMoERoutingShiftCases(value.Cases)
		return value
	},
)

var moeRoutingShiftLineage = func(value moeRoutingShiftEvidence) []artifact.Lineage {
	parents := []artifact.ID{value.BaselineMatrix}
	for _, testCase := range value.Cases {
		parents = append(parents, testCase.AggregateEvaluation)
		for _, segment := range testCase.Segments {
			parents = append(parents, segment.Stratum, segment.Evaluation)
			if segment.BiasSource != nil {
				parents = append(parents, *segment.BiasSource)
			}
		}
	}
	return artifact.DependencyLineage(value.ID, uniqueArtifactIDs(parents)...)
}

func canonicalizeMoERoutingShift(value *moeRoutingShiftEvidence) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.BaselineMatrix.Kind() != artifact.KindRecipe ||
		len(value.Cases) != len(moeRoutingShiftKinds) {
		return errors.New("evaluation: incomplete MoE routing shift evidence")
	}
	value.Cases = cloneMoERoutingShiftCases(value.Cases)
	slices.SortFunc(value.Cases, func(left, right moeRoutingShiftCase) int { return cmp.Compare(left.Kind, right.Kind) })
	required := make(map[moeRoutingShiftKind]bool, len(moeRoutingShiftKinds))
	for _, kind := range moeRoutingShiftKinds {
		required[kind] = false
	}
	hidden, lag := false, false
	for _, testCase := range value.Cases {
		if _, known := required[testCase.Kind]; !known || required[testCase.Kind] ||
			testCase.AggregateEvaluation.Kind() != artifact.KindEvidence || len(testCase.Segments) == 0 ||
			testCase.Kind == moeRoutingHomogeneous && len(testCase.Segments) != 1 ||
			testCase.Kind != moeRoutingHomogeneous && !hasComparisonPair(testCase.Segments) {
			return errors.New("evaluation: invalid MoE routing shift case")
		}
		required[testCase.Kind] = true
		seen := make(map[artifact.ID]bool, len(testCase.Segments))
		for _, segment := range testCase.Segments {
			if segment.Stratum.Kind() != artifact.KindDatasetShard || segment.Evaluation.Kind() != artifact.KindEvidence || seen[segment.Stratum] ||
				testCase.Kind == moeRoutingTransition && (segment.BiasSource == nil || segment.BiasSource.Kind() != artifact.KindDatasetShard) ||
				testCase.Kind != moeRoutingTransition && segment.BiasSource != nil {
				return errors.New("evaluation: invalid MoE routing shift segment")
			}
			seen[segment.Stratum] = true
			if (testCase.Kind == moeRoutingMixed || testCase.Kind == moeRoutingImbalanced) && testCase.AggregateAdmitted && !segment.Admitted {
				hidden = true
			}
			if testCase.Kind == moeRoutingTransition && *segment.BiasSource != segment.Stratum {
				lag = true
			}
		}
	}
	if value.HiddenStratumFailure != hidden || value.ControllerLag != lag {
		return errors.New("evaluation: MoE routing shift verdict differs from exact strata")
	}
	return nil
}

func cloneMoERoutingShiftCases(cases []moeRoutingShiftCase) []moeRoutingShiftCase {
	copy := slices.Clone(cases)
	for index := range copy {
		copy[index].Segments = slices.Clone(copy[index].Segments)
		for segment := range copy[index].Segments {
			if copy[index].Segments[segment].BiasSource != nil {
				copy[index].Segments[segment].BiasSource = artifact.IDPointer(*copy[index].Segments[segment].BiasSource)
			}
		}
	}
	return copy
}
