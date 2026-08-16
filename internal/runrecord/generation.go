package runrecord

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	GenerationVersion   uint16 = 1
	GenerationMediaType        = "application/vnd.overgo.generation+json"
	GenerationSchema           = "overgo/generation/v1"
	maxGenerationSeeds         = 64
)

// GenerationRecord is the experiment/generation aggregate: one descendant
// construction attempt, bound to the immutable identities needed to replay and
// attribute it. It stores references, never copies -- RepoDB stays generic
// storage, and descendant-generation depth is computed from the child/parent
// edges of this graph, never maintained in prose. Day-one schema is
// deliberately minimal; extend from what the first experiments demand.
type GenerationRecord struct {
	Version      uint16        `json:"version"`
	Parents      []artifact.ID `json:"parents"`
	Child        artifact.ID   `json:"child"`
	Components   []artifact.ID `json:"components,omitempty"`
	Bridge       artifact.ID   `json:"bridge,omitempty"`
	TrainingPlan artifact.ID   `json:"training_plan"`
	Dataset      artifact.ID   `json:"dataset"`
	Split        artifact.ID   `json:"split"`
	Evaluator    artifact.ID   `json:"evaluator"`
	Code         artifact.ID   `json:"code"`
	Environment  artifact.ID   `json:"environment"`
	Seeds        []uint64      `json:"seeds"`
	Budget       artifact.ID   `json:"budget"`
	Run          artifact.ID   `json:"run"`
	Outcome      Outcome       `json:"outcome"`
	Decision     artifact.ID   `json:"decision"`
	ID           artifact.ID   `json:"-"`
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

// The authoring constructor arrives with the experiment-lifecycle row, which
// supplies its production caller; until then generationCodec.New is reachable
// in-package and reading the graph is the only production capability.

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
	content, err := value.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, value.Lineage(), nil)
}

func canonicalizeGeneration(value *GenerationRecord) error {
	if value == nil || value.Version != GenerationVersion {
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
	if len(value.Seeds) == 0 || len(value.Seeds) > maxGenerationSeeds {
		return errors.New("run record: generation seeds are absent or unbounded")
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
