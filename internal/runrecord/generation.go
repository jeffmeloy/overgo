package runrecord

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	generationMediaType = "application/vnd.overgo.generation-record+json"
	generationSchema    = "overgo/generation-record/v1"
)

// GenerationRecord is the referential provenance root for one experiment.
// Its owners retain every underlying fact; this document binds their identities.
type GenerationRecord struct {
	Parents      []artifact.ID `json:"parents"`
	Components   []artifact.ID `json:"components"`
	Bridge       artifact.ID   `json:"bridge"`
	TrainingPlan artifact.ID   `json:"training_plan"`
	DataSplit    artifact.ID   `json:"data_split"`
	Evaluator    artifact.ID   `json:"evaluator"`
	Code         artifact.ID   `json:"code"`
	Kernel       artifact.ID   `json:"kernel"`
	Environment  artifact.ID   `json:"environment"`
	Seeds        artifact.ID   `json:"seeds"`
	Budget       artifact.ID   `json:"budget"`
	Run          artifact.ID   `json:"run"`
	Outcome      artifact.ID   `json:"outcome"`
	Decision     artifact.ID   `json:"decision"`
	ID           artifact.ID   `json:"-"`
}

var generationCodec = artifact.JSONDocumentCodec(
	"generation record", artifact.KindEvidence, generationMediaType, generationSchema, canonicalizeGeneration,
	func(value GenerationRecord) artifact.ID { return value.ID },
	func(value *GenerationRecord, id artifact.ID) { value.ID = id },
	func(value GenerationRecord) GenerationRecord {
		value.Parents = slices.Clone(value.Parents)
		value.Components = slices.Clone(value.Components)
		return value
	},
)

func NewGenerationRecord(value GenerationRecord) (GenerationRecord, error) {
	return generationCodec.New(value)
}

func (value GenerationRecord) Batch(key string) (artifact.Batch, error) {
	return artifact.DependencyDocumentBatch(key, generationCodec, value, generationDependencies(value)...)
}

func canonicalizeGeneration(value *GenerationRecord) error {
	if value == nil || len(value.Parents) == 0 || len(value.Components) == 0 {
		return errors.New("run record: generation provenance is incomplete")
	}
	if canonicalIDs(value.Parents) != nil || canonicalIDs(value.Components) != nil {
		return errors.New("run record: invalid generation component identity")
	}
	for _, parent := range value.Parents {
		if parent.Kind() != artifact.KindModel {
			return errors.New("run record: generation parent is not a model")
		}
	}
	if value.Bridge.Kind() != artifact.KindAdapter || value.TrainingPlan.Kind() != artifact.KindRecipe ||
		value.DataSplit.Kind() != artifact.KindDatasetShard || value.Evaluator.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Kernel.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || value.Seeds.Kind() != artifact.KindEvidence ||
		value.Budget.Kind() != artifact.KindEvidence || value.Run.Kind() != artifact.KindRun ||
		value.Outcome.Kind() != artifact.KindEvaluation || value.Decision.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid generation provenance")
	}
	if !distinctIDs(generationDependencies(*value)...) {
		return errors.New("run record: generation provenance identities are not distinct")
	}
	return nil
}

func generationDependencies(value GenerationRecord) []artifact.ID {
	return slices.Concat(value.Parents, value.Components, []artifact.ID{
		value.Bridge, value.TrainingPlan, value.DataSplit, value.Evaluator, value.Code, value.Kernel,
		value.Environment, value.Seeds, value.Budget, value.Run, value.Outcome, value.Decision,
	})
}
