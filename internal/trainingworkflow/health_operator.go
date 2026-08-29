package trainingworkflow

import (
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	healthOperatorMediaType = "application/vnd.overgo.training-health-operator-decision+json"
	healthOperatorSchema    = "overgo/training-health-operator-decision/v1"
)

type healthOperatorAction string

const (
	healthOperatorAdvisory    healthOperatorAction = "advisory"
	healthOperatorContainment healthOperatorAction = "containment"
	healthOperatorRetry       healthOperatorAction = "retry"
	healthOperatorRollback    healthOperatorAction = "rollback"
)

type healthOperatorDecision struct {
	ID             artifact.ID          `json:"-"`
	Version        uint16               `json:"version"`
	Run            artifact.ID          `json:"run"`
	Classification artifact.ID          `json:"classification"`
	HealthEvidence artifact.ID          `json:"health_evidence"`
	Policy         artifact.ID          `json:"policy"`
	Action         healthOperatorAction `json:"action"`
	AdvisoryOnly   bool                 `json:"advisory_only"`
	Experiment     artifact.ID          `json:"experiment,omitzero"`
	Lifecycle      artifact.ID          `json:"lifecycle,omitzero"`
}

var healthOperatorDecisionCodec = artifact.JSONDocumentCodec(
	"training health operator decision", artifact.KindEvidence,
	healthOperatorMediaType, healthOperatorSchema,
	func(value *healthOperatorDecision) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
			value.Classification.Kind() != artifact.KindEvidence || value.HealthEvidence.Kind() != artifact.KindEvidence ||
			value.Policy.Kind() != artifact.KindProfile {
			return errors.New("training workflow: invalid health operator decision authority")
		}
		if value.Action == healthOperatorAdvisory {
			if !value.AdvisoryOnly || value.Experiment.Valid() || value.Lifecycle.Valid() {
				return errors.New("training workflow: advisory health decision carries lifecycle mutation authority")
			}
			return nil
		}
		if value.AdvisoryOnly || value.Experiment.Kind() != artifact.KindEvidence || value.Lifecycle.Kind() != artifact.KindEvidence ||
			(value.Action != healthOperatorContainment && value.Action != healthOperatorRetry && value.Action != healthOperatorRollback) {
			return errors.New("training workflow: health mutation lacks lifecycle authority")
		}
		return nil
	},
	func(value healthOperatorDecision) artifact.ID { return value.ID },
	func(value *healthOperatorDecision, id artifact.ID) { value.ID = id }, nil,
)

var buildHealthOperatorDecision = func(
	classification healthClassification,
	action healthOperatorAction,
	lifecycle *runrecord.ExperimentLifecycle,
) (healthOperatorDecision, error) {
	if healthClassificationCodec.ValidateIdentity(classification) != nil {
		return healthOperatorDecision{}, errors.New("training workflow: health classification authority differs")
	}
	decision := healthOperatorDecision{
		Version: artifact.InitialDocumentVersion, Run: classification.Run, Classification: classification.ID,
		HealthEvidence: classification.HealthEvidence, Policy: classification.Policy, Action: action,
		AdvisoryOnly: action == healthOperatorAdvisory,
	}
	if action == healthOperatorAdvisory {
		if lifecycle != nil {
			return healthOperatorDecision{}, errors.New("training workflow: advisory decision does not consume lifecycle authority")
		}
		return healthOperatorDecisionCodec.New(decision)
	}
	if lifecycle == nil {
		return healthOperatorDecision{}, errors.New("training workflow: health evidence cannot mutate a run without lifecycle authority")
	}
	content, err := lifecycle.Content()
	if err != nil || content.Descriptor.ID != lifecycle.ID || lifecycle.Evidence != classification.ID {
		return healthOperatorDecision{}, errors.Join(err, errors.New("training workflow: lifecycle does not authorize this health classification"))
	}
	authorized := action == healthOperatorContainment && lifecycle.State == runrecord.ExperimentContained ||
		action == healthOperatorRollback && lifecycle.State == runrecord.ExperimentRolledBack ||
		action == healthOperatorRetry && lifecycle.State == runrecord.ExperimentLeased && lifecycle.Retry > 0
	if !authorized {
		return healthOperatorDecision{}, errors.New("training workflow: lifecycle state does not authorize the health action")
	}
	decision.Experiment, decision.Lifecycle = lifecycle.Experiment, lifecycle.ID
	return healthOperatorDecisionCodec.New(decision)
}

var healthOperatorDecisionLineage = func(value healthOperatorDecision) []artifact.Lineage {
	parents := []artifact.ID{value.Run, value.Classification, value.HealthEvidence, value.Policy}
	if value.Experiment.Valid() {
		parents = append(parents, value.Experiment, value.Lifecycle)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}
