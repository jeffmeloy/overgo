package trainingdata

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	evaluationIsolationMediaType = "application/vnd.overgo.evaluation-isolation+json"
	evaluationIsolationSchema    = "overgo/evaluation-isolation/v1"
)

type SplitBinding struct {
	Split  artifact.ID `json:"split"`
	Budget artifact.ID `json:"budget"`
}

// EvaluationIsolation binds four disjoint data surfaces to immutable budgets.
type EvaluationIsolation struct {
	Dataset     artifact.ID  `json:"dataset"`
	Development SplitBinding `json:"development"`
	Selection   SplitBinding `json:"selection"`
	Promotion   SplitBinding `json:"promotion"`
	Audit       SplitBinding `json:"audit"`
	ID          artifact.ID  `json:"-"`
}

var evaluationIsolationCodec = artifact.JSONDocumentCodec(
	"evaluation isolation", artifact.KindEvidence, evaluationIsolationMediaType, evaluationIsolationSchema,
	canonicalizeEvaluationIsolation,
	func(value EvaluationIsolation) artifact.ID { return value.ID },
	func(value *EvaluationIsolation, id artifact.ID) { value.ID = id }, nil,
)

func NewEvaluationIsolation(value EvaluationIsolation) (EvaluationIsolation, error) {
	return evaluationIsolationCodec.New(value)
}

func (value EvaluationIsolation) BudgetFor(split artifact.ID) (artifact.ID, error) {
	if err := evaluationIsolationCodec.ValidateIdentity(value); err != nil {
		return artifact.ID{}, err
	}
	for _, binding := range value.bindings() {
		if binding.Split == split {
			return binding.Budget, nil
		}
	}
	return artifact.ID{}, errors.New("training data: split is outside evaluation isolation")
}

// AuthorizeProposer keeps promotion and audit results outside proposal input.
func (value EvaluationIsolation) AuthorizeProposer(split artifact.ID) error {
	if err := evaluationIsolationCodec.ValidateIdentity(value); err != nil {
		return err
	}
	if split != value.Development.Split && split != value.Selection.Split {
		return errors.New("training data: proposer cannot observe held-out split")
	}
	return nil
}

func (value EvaluationIsolation) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{value.Dataset}
	for _, binding := range value.bindings() {
		parents = append(parents, binding.Split, binding.Budget)
	}
	return artifact.DependencyDocumentBatch(key, evaluationIsolationCodec, value, parents...)
}

func (value EvaluationIsolation) bindings() []SplitBinding {
	return []SplitBinding{value.Development, value.Selection, value.Promotion, value.Audit}
}

func canonicalizeEvaluationIsolation(value *EvaluationIsolation) error {
	if value == nil || value.Dataset.Kind() != artifact.KindDataset {
		return errors.New("training data: invalid evaluation isolation")
	}
	splits, budgets := map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for _, binding := range value.bindings() {
		if binding.Split.Kind() != artifact.KindDatasetShard || binding.Budget.Kind() != artifact.KindEvidence ||
			splits[binding.Split] || budgets[binding.Budget] {
			return errors.New("training data: evaluation splits and budgets must be typed and distinct")
		}
		splits[binding.Split], budgets[binding.Budget] = true, true
	}
	return nil
}
