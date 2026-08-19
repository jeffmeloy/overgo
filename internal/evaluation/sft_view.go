package evaluation

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

const (
	sftEvaluationViewVersion   uint16 = 1
	sftEvaluationViewMediaType        = "application/vnd.overgo.sft-evaluation-view+json"
	sftEvaluationViewSchema           = "overgo/sft-evaluation-view/v1"
)

var sftEvaluationViewCodec = artifact.JSONDocumentCodec(
	"SFT evaluation view", artifact.KindProfile, sftEvaluationViewMediaType, sftEvaluationViewSchema,
	canonicalizeSFTEvaluationView, func(value SFTEvaluationView) artifact.ID { return value.ID },
	func(value *SFTEvaluationView, id artifact.ID) { value.ID = id }, cloneSFTEvaluationView,
)

type SFTEvaluationView struct {
	ID                 artifact.ID                      `json:"-"`
	Version            uint16                           `json:"version"`
	Objective          artifact.ID                      `json:"objective"`
	Dataset            artifact.ID                      `json:"dataset"`
	TrainingMembership artifact.ID                      `json:"training_membership"`
	HeldoutMembership  artifact.ID                      `json:"heldout_membership"`
	Processors         []artifact.ID                    `json:"processors"`
	Projectors         []artifact.ID                    `json:"projectors,omitempty"`
	Codecs             []artifact.ID                    `json:"codecs,omitempty"`
	Signature          recipecontract.ModalitySignature `json:"signature"`
	Records            []dataset.Record                 `json:"records"`
}

func CompileSFTEvaluationView(
	objective trainingprogram.ObjectiveDocument,
	training, heldout dataset.Membership,
) (SFTEvaluationView, error) {
	if _, err := objective.Content(); err != nil {
		return SFTEvaluationView{}, err
	}
	if err := training.ValidateIdentity(); err != nil {
		return SFTEvaluationView{}, err
	}
	if err := heldout.ValidateIdentity(); err != nil {
		return SFTEvaluationView{}, err
	}
	if objective.Split != training.ID || objective.Dataset != training.Source || heldout.Source != objective.Dataset ||
		heldout.ID == training.ID || len(heldout.Records) == 0 {
		return SFTEvaluationView{}, errors.New("evaluation: held-out membership differs from training authority")
	}
	trainingRecords := make(map[string]struct{}, len(training.Records))
	trainingGroups := make(map[string]struct{}, len(training.Records))
	for _, record := range training.Records {
		trainingRecords[record.ID] = struct{}{}
		trainingGroups[record.Group] = struct{}{}
	}
	for _, record := range heldout.Records {
		if _, overlap := trainingRecords[record.ID]; overlap {
			return SFTEvaluationView{}, errors.New("evaluation: held-out record occurs in training membership")
		}
		if _, overlap := trainingGroups[record.Group]; overlap {
			return SFTEvaluationView{}, errors.New("evaluation: held-out group occurs in training membership")
		}
	}
	return sftEvaluationViewCodec.New(SFTEvaluationView{
		Version: sftEvaluationViewVersion, Objective: objective.ID, Dataset: objective.Dataset,
		TrainingMembership: training.ID, HeldoutMembership: heldout.ID,
		Processors: slices.Clone(objective.Processors), Projectors: slices.Clone(objective.Projectors),
		Codecs: slices.Clone(objective.Codecs), Signature: objective.Signature.Clone(),
		Records: slices.Clone(heldout.Records),
	})
}

func (view SFTEvaluationView) Content() (artifact.Content, error) {
	return sftEvaluationViewCodec.Content(view)
}

func (view SFTEvaluationView) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{
		view.Objective, view.Dataset, view.TrainingMembership, view.HeldoutMembership,
	}
	parents = append(parents, view.Processors...)
	parents = append(parents, view.Projectors...)
	parents = append(parents, view.Codecs...)
	return sftEvaluationViewCodec.Batch(key, view, artifact.DependencyLineage(view.ID, parents...), nil)
}

func canonicalizeSFTEvaluationView(view *SFTEvaluationView) error {
	if view == nil || view.Version != sftEvaluationViewVersion || view.Objective.Kind() != artifact.KindProfile ||
		view.Dataset.Kind() != artifact.KindDataset || view.TrainingMembership.Kind() != artifact.KindDatasetShard ||
		view.HeldoutMembership.Kind() != artifact.KindDatasetShard || view.TrainingMembership == view.HeldoutMembership ||
		len(view.Processors) == 0 || len(view.Records) == 0 || view.Signature.Validate() != nil {
		return errors.New("evaluation: invalid SFT evaluation view")
	}
	for _, declaration := range []struct {
		ids  []artifact.ID
		kind artifact.Kind
	}{{view.Processors, artifact.KindProfile}, {view.Projectors, artifact.KindProjector}, {view.Codecs, artifact.KindProfile}} {
		for index, id := range declaration.ids {
			if id.Kind() != declaration.kind || index > 0 && declaration.ids[index-1] == id {
				return errors.New("evaluation: invalid SFT component identity")
			}
		}
	}
	for index, record := range view.Records {
		if record.ID == "" || record.Group == "" || index > 0 && view.Records[index-1].ID >= record.ID {
			return errors.New("evaluation: invalid SFT held-out record")
		}
	}
	return nil
}

func cloneSFTEvaluationView(view SFTEvaluationView) SFTEvaluationView {
	view.Processors = slices.Clone(view.Processors)
	view.Projectors = slices.Clone(view.Projectors)
	view.Codecs = slices.Clone(view.Codecs)
	view.Signature = view.Signature.Clone()
	view.Records = slices.Clone(view.Records)
	return view
}
