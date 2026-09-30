package runrecord

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	trainingDecisionMediaType = "application/vnd.overgo.training-decision+json"
	trainingDecisionSchema    = "overgo/training-decision/v1"
)

type trainingDecisionState string

const (
	// TrainingEvaluationRequired withholds promotion until an evaluation
	// judges the checkpoint.
	TrainingEvaluationRequired trainingDecisionState = "evaluation-required"
	// trainingApproved records a held-out verdict that the checkpoint beat
	// its parent on records its training never saw.
	trainingApproved trainingDecisionState = "approved"
	// trainingRefused records a held-out verdict that found no gain: the
	// parent stays, as the rollback names.
	trainingRefused trainingDecisionState = "refused"
)

type TrainingDecision struct {
	Version    uint16                `json:"version"`
	State      trainingDecisionState `json:"state"`
	Run        artifact.ID           `json:"run"`
	Recipe     artifact.ID           `json:"recipe"`
	Parent     artifact.ID           `json:"parent"`
	Checkpoint artifact.ID           `json:"checkpoint"`
	Trace      artifact.ID           `json:"trace"`
	Rollback   artifact.ID           `json:"rollback"`
	// Verdict is the held-out verdict an approved or refused decision
	// stands on.
	Verdict artifact.ID `json:"verdict,omitzero"`
	ID      artifact.ID `json:"-"`
}

var trainingDecisionCodec = artifact.JSONDocumentCodec(
	"training decision", artifact.KindEvidence, trainingDecisionMediaType, trainingDecisionSchema,
	func(value *TrainingDecision) error {
		judged := value != nil && (value.State == trainingApproved || value.State == trainingRefused)
		if value == nil || value.Version != artifact.InitialDocumentVersion ||
			value.State != TrainingEvaluationRequired && !judged || judged != (value.Verdict.Kind() == artifact.KindEvaluation) ||
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
	return trainingDecisionCodec.NewInitial(TrainingDecision{
		State: TrainingEvaluationRequired,
		Run:   run, Recipe: recipeID, Parent: parent, Checkpoint: checkpoint, Trace: trace, Rollback: parent,
	})
}

// Judged settles the decision on a held-out verdict: approved when it
// passed, refused when it did not.
func (decision TrainingDecision) Judged(verdict artifact.ID, passed bool) (TrainingDecision, error) {
	decision.State, decision.Verdict = trainingRefused, verdict
	if passed {
		decision.State = trainingApproved
	}
	return trainingDecisionCodec.New(decision)
}

func (decision TrainingDecision) Content() (artifact.Content, error) {
	return trainingDecisionCodec.Content(decision)
}

func (decision TrainingDecision) Lineage() []artifact.Lineage {
	parents := []artifact.ID{decision.Run, decision.Recipe, decision.Parent, decision.Checkpoint, decision.Trace}
	if decision.Verdict.Valid() {
		parents = append(parents, decision.Verdict)
	}
	return artifact.DependencyLineage(decision.ID, parents...)
}
