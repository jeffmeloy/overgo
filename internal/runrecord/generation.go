package runrecord

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
)

const (
	GenerationMediaType = "application/vnd.overgo.generation+json"
	GenerationSchema    = "overgo/generation/v1"
)

// GenerationRecord is the experiment/generation aggregate: one descendant
// construction attempt, bound to the immutable identities needed to replay and
// attribute it. It stores references, never copies -- OvergoDB stays generic
// storage, and descendant-generation depth is computed from the child/parent
// edges of this graph, never maintained in prose. Day-one schema is
// deliberately minimal; extend from what the first experiments demand.
type GenerationRecord struct {
	Version    uint16        `json:"version"`
	Parents    []artifact.ID `json:"parents"`
	Child      artifact.ID   `json:"child"`
	Components []artifact.ID `json:"components,omitempty"`
	// omitzero, not omitempty: a whole-model child legally carries no bridge,
	// and the zero ID must vanish from the document rather than fail the
	// strict ID marshal.
	Bridge       artifact.ID `json:"bridge,omitzero"`
	TrainingPlan artifact.ID `json:"training_plan"`
	Dataset      artifact.ID `json:"dataset"`
	Split        artifact.ID `json:"split"`
	Evaluator    artifact.ID `json:"evaluator"`
	Code         artifact.ID `json:"code"`
	Environment  artifact.ID `json:"environment"`
	Seeds        []uint64    `json:"seeds"`
	Budget       artifact.ID `json:"budget"`
	Run          artifact.ID `json:"run"`
	Outcome      Outcome     `json:"outcome"`
	Decision     artifact.ID `json:"decision"`
	ID           artifact.ID `json:"-"`
}

var generationCodec = artifact.JSONDocumentCodec(
	"generation record", artifact.KindEvidence, GenerationMediaType, GenerationSchema,
	canonicalizeGeneration,
	func(value GenerationRecord) artifact.ID { return value.ID },
	func(value *GenerationRecord, id artifact.ID) { value.ID = id },
	func(value GenerationRecord) GenerationRecord {
		value.Parents = slices.Clone(value.Parents)
		value.Components = slices.Clone(value.Components)
		value.Seeds = slices.Clone(value.Seeds)
		return value
	},
)

// NewGenerationRecord identifies one descendant construction attempt. Its
// production caller is composition.RecordViability (the graft probe's
// recording path), which arrived with the generation-lifecycle wiring.
func NewGenerationRecord(record GenerationRecord) (GenerationRecord, error) {
	record.Version = artifact.InitialDocumentVersion
	record.ID = artifact.ID{}
	return generationCodec.New(record)
}

// ValidateGenerationBudget proves a record's seed consumption fits its bound
// grant: the budget document must be the one the record references, and the
// seed count may not exceed the issued amount. The bound replaces the former
// maxGenerationSeeds code threshold with the typed experiment budget.
func ValidateGenerationBudget(record GenerationRecord, budget Budget) error {
	if err := budget.ValidateIdentity(); err != nil {
		return err
	}
	if budget.ID != record.Budget {
		return fmt.Errorf("run record: generation %s is bound to budget %s, not %s", record.ID, record.Budget, budget.ID)
	}
	if uint64(len(record.Seeds)) > budget.Issued {
		return fmt.Errorf("run record: generation %s consumed %d seeds against a grant of %d %s",
			record.ID, len(record.Seeds), budget.Issued, budget.Unit)
	}
	return nil
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

// Lineage binds the child to every parent (the depth-bearing edges) and the
// record itself to every referenced identity, so the experiment graph is
// queryable without decoding record bodies.
func (value GenerationRecord) Lineage() []artifact.Lineage {
	dependencies := []artifact.ID{value.Child, value.TrainingPlan, value.Dataset, value.Split,
		value.Evaluator, value.Code, value.Environment, value.Budget, value.Run, value.Decision}
	dependencies = append(dependencies, value.Parents...)
	dependencies = append(dependencies, value.Components...)
	if value.Bridge.Valid() {
		dependencies = append(dependencies, value.Bridge)
	}
	lineage := artifact.DependencyLineage(value.ID, dependencies...)
	for _, parent := range value.Parents {
		lineage = append(lineage, artifact.Lineage{
			Child: value.Child, Parent: parent, Relation: artifact.RelationTrainedFrom,
		})
	}
	return lineage
}

func (value GenerationRecord) Batch(key string) (artifact.Batch, error) {
	return generationCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeGeneration(value *GenerationRecord) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid generation record version")
	}
	if len(value.Parents) == 0 || !distinctIDs(value.Parents...) {
		return errors.New("run record: generation record needs distinct parents")
	}
	for _, parent := range value.Parents {
		if parent.Kind() != artifact.KindModel {
			return errors.New("run record: generation parent is not a model")
		}
	}
	if value.Child.Kind() != artifact.KindModel || slices.Contains(value.Parents, value.Child) {
		return errors.New("run record: generation child must be a model distinct from its parents")
	}
	// Components and bridge arrive together: a graft without its trained
	// adapter is not replayable, and an adapter without components binds
	// nothing. A from-scratch or whole-model child carries neither.
	if (len(value.Components) == 0) != !value.Bridge.Valid() {
		return errors.New("run record: components and bridge are both-or-neither")
	}
	if !distinctIDs(append(slices.Clone(value.Components), value.Parents...)...) && len(value.Components) > 0 {
		return errors.New("run record: generation components repeat")
	}
	if value.Bridge.Valid() && value.Bridge.Kind() != artifact.KindAdapter {
		return errors.New("run record: generation bridge is not an adapter")
	}
	// Seed count is bounded by the record's typed budget grant, never by a
	// code threshold; ValidateGenerationBudget enforces it where the budget
	// document is available.
	if len(value.Seeds) == 0 {
		return errors.New("run record: generation seeds are absent")
	}
	if value.TrainingPlan.Kind() != artifact.KindRecipe || value.Dataset.Kind() != artifact.KindDataset ||
		value.Split.Kind() != artifact.KindDatasetShard || value.Evaluator.Kind() != artifact.KindEvidence ||
		value.Code.Kind() != artifact.KindEvidence || value.Environment.Kind() != artifact.KindEvidence ||
		value.Budget.Kind() != artifact.KindEvidence || value.Run.Kind() != artifact.KindRun ||
		value.Decision.Kind() != artifact.KindEvidence {
		return errors.New("run record: generation reference kind mismatch")
	}
	if !distinctIDs(value.Evaluator, value.Code, value.Environment, value.Budget, value.Decision) {
		return errors.New("run record: generation evidence references repeat")
	}
	switch value.Outcome {
	case OutcomeSucceeded, OutcomeFailed, OutcomeCancelled:
	default:
		return errors.New("run record: invalid generation outcome")
	}
	return nil
}
