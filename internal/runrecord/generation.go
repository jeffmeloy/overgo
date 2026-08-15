package runrecord

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
)

const (
	GenerationMediaType = "application/vnd.overgo.generation-record+json"
	GenerationSchema    = "overgo/generation-record/v1"
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
	"generation record", artifact.KindEvidence, GenerationMediaType, GenerationSchema, canonicalizeGeneration,
	func(value GenerationRecord) artifact.ID { return value.ID },
	func(value *GenerationRecord, id artifact.ID) { value.ID = id },
	func(value GenerationRecord) GenerationRecord {
		value.Parents = slices.Clone(value.Parents)
		value.Components = slices.Clone(value.Components)
		return value
	},
)

func NewGenerationRecord(value GenerationRecord) (GenerationRecord, error) {
	value.ID = artifact.ID{}
	return generationCodec.New(value)
}

func ParseGenerationRecord(data []byte) (GenerationRecord, error) {
	return generationCodec.Parse(data)
}

func (value GenerationRecord) ValidateIdentity() error {
	return generationCodec.ValidateIdentity(value)
}

func (value GenerationRecord) Content() (artifact.Content, error) {
	return generationCodec.Content(value)
}

func (value GenerationRecord) Lineage() []artifact.Lineage {
	parents := slices.Concat(value.Parents, value.Components, []artifact.ID{
		value.Bridge, value.TrainingPlan, value.DataSplit, value.Evaluator, value.Code, value.Kernel,
		value.Environment, value.Seeds, value.Budget, value.Run, value.Outcome, value.Decision,
	})
	return dependencyLineage(value.ID, parents...)
}

func (value GenerationRecord) Batch(key string) (artifact.Batch, error) {
	content, err := value.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, value.Lineage(), nil)
}

func canonicalizeGeneration(value *GenerationRecord) error {
	if value == nil || !canonicalGenerationIDs(value.Parents, artifact.KindModel) ||
		!canonicalGenerationIDs(value.Components, artifact.KindInvalid) ||
		value.Bridge.Kind() != artifact.KindAdapter || value.TrainingPlan.Kind() != artifact.KindRecipe ||
		value.DataSplit.Kind() != artifact.KindDatasetShard || value.Evaluator.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Kernel.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || value.Seeds.Kind() != artifact.KindEvidence ||
		value.Budget.Kind() != artifact.KindEvidence || value.Run.Kind() != artifact.KindRun ||
		value.Outcome.Kind() != artifact.KindEvaluation || value.Decision.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid generation provenance")
	}
	identities := slices.Concat(value.Parents, value.Components, []artifact.ID{
		value.Bridge, value.TrainingPlan, value.DataSplit, value.Evaluator, value.Code, value.Kernel,
		value.Environment, value.Seeds, value.Budget, value.Run, value.Outcome, value.Decision,
	})
	if !distinctIDs(identities...) {
		return errors.New("run record: generation provenance identities are not distinct")
	}
	return nil
}

func canonicalGenerationIDs(values []artifact.ID, kind artifact.Kind) bool {
	if len(values) == 0 {
		return false
	}
	for _, value := range values {
		if !value.Valid() || kind != artifact.KindInvalid && value.Kind() != kind {
			return false
		}
	}
	sort.Slice(values, func(i, j int) bool { return values[i].String() < values[j].String() })
	return true
}
