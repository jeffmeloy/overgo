package runrecord

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// MoERouterObservationMediaType identifies exact per-layer router observations.
	MoERouterObservationMediaType = "application/vnd.overgo.moe-router-observation+json"
	// MoERouterObservationSchema identifies the router observation contract version.
	MoERouterObservationSchema = "overgo/moe-router-observation/v1"
)

// MoERouterMargin preserves observed versus unknown selected-boundary margins.
type MoERouterMargin struct {
	Observed bool    `json:"observed"`
	Value    float32 `json:"value"`
}

// MoERouterObservation records exact routing choices for one layer and training
// step. Selections and combine weights are row-major in router rank order;
// Accepted distinguishes capacity drops from selected work. An observed margin
// is the activated score of the final selected expert minus the greatest
// unselected score for that row. Unknown margins remain explicit.
type MoERouterObservation struct {
	Version        uint16            `json:"version"`
	Run            artifact.ID       `json:"run"`
	Model          artifact.ID       `json:"model"`
	Dataset        artifact.ID       `json:"dataset"`
	Split          artifact.ID       `json:"split"`
	Recipe         artifact.ID       `json:"recipe"`
	Code           artifact.ID       `json:"code"`
	Checkpoint     artifact.ID       `json:"checkpoint"`
	Policy         artifact.ID       `json:"policy"`
	Step           uint64            `json:"step"`
	Layer          uint32            `json:"layer"`
	Rows           uint64            `json:"rows"`
	Experts        uint32            `json:"experts"`
	TopK           uint32            `json:"top_k"`
	Selections     []uint32          `json:"selections"`
	CombineWeights []float32         `json:"combine_weights"`
	Accepted       []bool            `json:"accepted"`
	Margins        []MoERouterMargin `json:"margins"`
	ID             artifact.ID       `json:"-"`
}

var moeRouterObservationCodec = artifact.JSONDocumentCodec(
	"moe router observation", artifact.KindEvidence,
	MoERouterObservationMediaType, MoERouterObservationSchema,
	canonicalizeMoERouterObservation,
	func(value MoERouterObservation) artifact.ID { return value.ID },
	func(value *MoERouterObservation, id artifact.ID) { value.ID = id },
	func(value MoERouterObservation) MoERouterObservation {
		value.Selections = slices.Clone(value.Selections)
		value.CombineWeights = slices.Clone(value.CombineWeights)
		value.Accepted = slices.Clone(value.Accepted)
		value.Margins = slices.Clone(value.Margins)
		return value
	},
)

// NewMoERouterObservation validates and identifies one exact layer observation.
func NewMoERouterObservation(value MoERouterObservation) (MoERouterObservation, error) {
	value.Version = artifact.InitialDocumentVersion
	moeRouterObservationCodec.SetIdentity(&value, artifact.ID{})
	return moeRouterObservationCodec.New(value)
}

func (value MoERouterObservation) content() (artifact.Content, error) {
	return moeRouterObservationCodec.Content(value)
}

func canonicalizeMoERouterObservation(value *MoERouterObservation) error {
	if value != nil {
		value.Rows = uint64(len(value.Margins))
	}
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Run.Kind() != artifact.KindRun || value.Model.Kind() != artifact.KindModel ||
		value.Dataset.Kind() != artifact.KindDataset || value.Split.Kind() != artifact.KindDatasetShard ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Code.Kind() != artifact.KindEvidence ||
		value.Checkpoint.Kind() != artifact.KindCheckpoint || value.Policy.Kind() != artifact.KindRecipe ||
		value.Rows == 0 || value.Experts == 0 || value.TopK == 0 || value.TopK > value.Experts {
		return errors.New("run record: invalid moe router observation scope")
	}
	selectionCount, ok := checked.Mul64(value.Rows, uint64(value.TopK))
	if !ok || selectionCount > uint64(^uint(0)>>1) || len(value.Selections) != int(selectionCount) ||
		len(value.CombineWeights) != int(selectionCount) || len(value.Accepted) != int(selectionCount) {
		return errors.New("run record: moe router observation shape differs from rows and top-k")
	}
	stride := int(value.TopK)
	for start := 0; start < len(value.Selections); start += stride {
		rowSelections := value.Selections[start : start+stride]
		for offset, expert := range rowSelections {
			weight := value.CombineWeights[start+offset]
			if expert >= value.Experts || !checked.Finite32(weight) {
				return errors.New("run record: invalid moe router selection")
			}
			for _, prior := range rowSelections[:offset] {
				if prior == expert {
					return errors.New("run record: duplicate moe expert selection")
				}
			}
		}
	}
	for _, margin := range value.Margins {
		if !margin.Observed && margin.Value != 0 || margin.Observed && (margin.Value < 0 || !checked.Finite32(margin.Value)) {
			return errors.New("run record: invalid moe router margin")
		}
	}
	return nil
}
