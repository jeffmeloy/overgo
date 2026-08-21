package runrecord

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	TrainingDecisionMediaType = "application/vnd.overgo.training-decision+json"
	TrainingDecisionSchema    = "overgo/training-decision/v1"
)

type TrainingDecisionState string

const TrainingEvaluationRequired TrainingDecisionState = "evaluation-required"

type TrainingDecision struct {
	Version    uint16                `json:"version"`
	State      TrainingDecisionState `json:"state"`
	Run        artifact.ID           `json:"run"`
	Recipe     artifact.ID           `json:"recipe"`
	Parent     artifact.ID           `json:"parent"`
	Checkpoint artifact.ID           `json:"checkpoint"`
	Trace      artifact.ID           `json:"trace"`
	Rollback   artifact.ID           `json:"rollback"`
	ID         artifact.ID           `json:"-"`
}

var trainingDecisionCodec = artifact.JSONDocumentCodec(
	"training decision", artifact.KindEvidence, TrainingDecisionMediaType, TrainingDecisionSchema,
	func(value *TrainingDecision) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.State != TrainingEvaluationRequired ||
			value.Run.Kind() != artifact.KindRun || value.Recipe.Kind() != artifact.KindRecipe ||
			value.Parent.Kind() != artifact.KindModel || value.Checkpoint.Kind() != artifact.KindCheckpoint ||
			value.Trace.Kind() != artifact.KindEvidence || value.Rollback != value.Parent {
			return errors.New("training decision: invalid authority")
		}
		return nil
	},
	func(value TrainingDecision) artifact.ID { return value.ID },
	func(value *TrainingDecision, id artifact.ID) { value.ID = id }, nil,
)

func NewTrainingDecision(run, recipeID, parent, checkpoint, trace artifact.ID) (TrainingDecision, error) {
	return trainingDecisionCodec.New(TrainingDecision{
		Version: artifact.InitialDocumentVersion, State: TrainingEvaluationRequired,
		Run: run, Recipe: recipeID, Parent: parent, Checkpoint: checkpoint, Trace: trace, Rollback: parent,
	})
}

func (decision TrainingDecision) Content() (artifact.Content, error) {
	return trainingDecisionCodec.Content(decision)
}

func (decision TrainingDecision) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(decision.ID, decision.Run, decision.Recipe, decision.Parent, decision.Checkpoint, decision.Trace)
}
