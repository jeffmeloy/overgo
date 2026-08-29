package trainingworkflow

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	healthClassificationMediaType = "application/vnd.overgo.training-health-classification+json"
	healthClassificationSchema    = "overgo/training-health-classification/v1"
)

type healthClassifierKind string

const (
	healthClassifierAbsentTelemetry     healthClassifierKind = "absent-telemetry"
	healthClassifierStall               healthClassifierKind = "stall"
	healthClassifierNonfinite           healthClassifierKind = "nonfinite"
	healthClassifierSustainedRegression healthClassifierKind = "sustained-regression"
)

var healthClassifierKinds = [...]healthClassifierKind{
	healthClassifierAbsentTelemetry, healthClassifierStall, healthClassifierNonfinite, healthClassifierSustainedRegression,
}

type healthClassifierFact struct {
	Kind                healthClassifierKind `json:"kind"`
	Signal              artifact.ID          `json:"signal"`
	Phase               artifact.ID          `json:"phase"`
	Mixture             artifact.ID          `json:"mixture"`
	Observation         artifact.ID          `json:"observation"`
	Baseline            artifact.ID          `json:"baseline"`
	Method              artifact.ID          `json:"method"`
	WindowEvidence      artifact.ID          `json:"window_evidence"`
	WindowAdmitted      bool                 `json:"window_admitted"`
	PersistenceEvidence artifact.ID          `json:"persistence_evidence"`
	PersistenceAdmitted bool                 `json:"persistence_admitted"`
	BoundaryEvidence    artifact.ID          `json:"boundary_evidence"`
	ReopenTrigger       artifact.ID          `json:"reopen_trigger"`
	Observed            bool                 `json:"observed"`
	KnownZero           bool                 `json:"known_zero"`
	Finite              bool                 `json:"finite"`
	Advanced            bool                 `json:"advanced"`
	RobustRegression    bool                 `json:"robust_regression"`
	Triggered           bool                 `json:"triggered"`
}

type healthClassification struct {
	ID             artifact.ID            `json:"-"`
	Version        uint16                 `json:"version"`
	Run            artifact.ID            `json:"run"`
	HealthEvidence artifact.ID            `json:"health_evidence"`
	Policy         artifact.ID            `json:"policy"`
	Facts          []healthClassifierFact `json:"facts"`
}

var healthClassificationCodec = artifact.JSONDocumentCodec(
	"training health classification", artifact.KindEvidence,
	healthClassificationMediaType, healthClassificationSchema,
	canonicalizeHealthClassification,
	func(value healthClassification) artifact.ID { return value.ID },
	func(value *healthClassification, id artifact.ID) { value.ID = id },
	func(value healthClassification) healthClassification {
		value.Facts = slices.Clone(value.Facts)
		return value
	},
)

var healthClassificationLineage = func(value healthClassification) []artifact.Lineage {
	parents := []artifact.ID{value.Run, value.HealthEvidence, value.Policy}
	for _, fact := range value.Facts {
		parents = append(parents,
			fact.Signal, fact.Phase, fact.Mixture, fact.Observation, fact.Baseline, fact.Method,
			fact.WindowEvidence, fact.PersistenceEvidence, fact.BoundaryEvidence, fact.ReopenTrigger,
		)
	}
	seen := make(map[artifact.ID]bool, len(parents))
	unique := make([]artifact.ID, 0, len(parents))
	for _, parent := range parents {
		if !seen[parent] {
			seen[parent] = true
			unique = append(unique, parent)
		}
	}
	return artifact.DependencyLineage(value.ID, unique...)
}

func canonicalizeHealthClassification(value *healthClassification) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun || value.HealthEvidence.Kind() != artifact.KindEvidence ||
		value.Policy.Kind() != artifact.KindProfile || len(value.Facts) != len(healthClassifierKinds) {
		return errors.New("training workflow: incomplete health classification")
	}
	value.Facts = slices.Clone(value.Facts)
	slices.SortFunc(value.Facts, func(left, right healthClassifierFact) int {
		return cmp.Compare(left.Kind, right.Kind)
	})
	required := make(map[healthClassifierKind]bool, len(healthClassifierKinds))
	for _, kind := range healthClassifierKinds {
		required[kind] = false
	}
	phase, mixture, boundary := value.Facts[0].Phase, value.Facts[0].Mixture, value.Facts[0].BoundaryEvidence
	for _, fact := range value.Facts {
		if _, known := required[fact.Kind]; !known || required[fact.Kind] ||
			fact.Signal.Kind() != artifact.KindProfile || fact.Phase.Kind() != artifact.KindProfile ||
			fact.Mixture.Kind() != artifact.KindDatasetShard || fact.Observation.Kind() != artifact.KindEvidence ||
			fact.Baseline.Kind() != artifact.KindEvidence || fact.Method.Kind() != artifact.KindProfile ||
			fact.WindowEvidence.Kind() != artifact.KindEvidence || fact.PersistenceEvidence.Kind() != artifact.KindEvidence ||
			fact.BoundaryEvidence.Kind() != artifact.KindEvidence || fact.ReopenTrigger.Kind() != artifact.KindRecipe ||
			fact.Phase != phase || fact.Mixture != mixture || fact.BoundaryEvidence != boundary ||
			fact.KnownZero && (!fact.Observed || !fact.Finite) {
			return errors.New("training workflow: invalid or mixed-boundary health classifier fact")
		}
		required[fact.Kind] = true
		triggered := false
		switch fact.Kind {
		case healthClassifierAbsentTelemetry:
			if !fact.Observed && (fact.KnownZero || fact.Finite) || fact.Advanced || fact.RobustRegression {
				return errors.New("training workflow: absent telemetry fact carries observed-value claims")
			}
			triggered = !fact.Observed && fact.WindowAdmitted && fact.PersistenceAdmitted
		case healthClassifierStall:
			if !fact.Observed || fact.KnownZero || fact.Finite || fact.RobustRegression {
				return errors.New("training workflow: stall fact lacks progress telemetry")
			}
			triggered = !fact.Advanced && fact.WindowAdmitted && fact.PersistenceAdmitted
		case healthClassifierNonfinite:
			if !fact.Observed || fact.Advanced || fact.RobustRegression {
				return errors.New("training workflow: nonfinite fact lacks numeric telemetry")
			}
			triggered = !fact.Finite
		case healthClassifierSustainedRegression:
			if !fact.Observed || !fact.Finite || fact.Advanced {
				return errors.New("training workflow: regression fact lacks finite metric evidence")
			}
			triggered = fact.RobustRegression && fact.WindowAdmitted && fact.PersistenceAdmitted
		}
		if fact.Triggered != triggered {
			return errors.New("training workflow: health verdict differs from exact classifier evidence")
		}
	}
	return nil
}
