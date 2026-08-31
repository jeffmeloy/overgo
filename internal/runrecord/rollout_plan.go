package runrecord

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// RolloutPlanMediaType identifies rollout plan documents.
	RolloutPlanMediaType = "application/vnd.overgo.rollout-plan+json"
	// RolloutPlanSchema identifies the rollout plan contract.
	RolloutPlanSchema = "overgo/rollout-plan/v1"
	// RolloutHashVersion is the registered deterministic cohort-assignment
	// hash version; the stable-assignment owner implements exactly this
	// version, so a plan cannot cite an assignment scheme that does not
	// exist.
	RolloutHashVersion = 1
	// RolloutCohortDenominator expresses cohort size in basis points: a
	// derived bound over 10000 keeps the share exact without floats.
	RolloutCohortDenominator = 10000
	// rolloutTextBytes bounds every free-text plan field.
	rolloutTextBytes = 256
)

// RolloutObservation is the observation contract one rollout must satisfy
// before its boundary decision: the metric observed on the candidate
// cohort, the minimum observation count, and the evidence the requirement
// derives from — never a hand-picked number.
type RolloutObservation struct {
	Metric          string      `json:"metric"`
	MinObservations uint64      `json:"min_observations"`
	Evidence        artifact.ID `json:"evidence"`
}

// RolloutPlan is an immutable, content-addressed promotion policy — not a
// mutable feature flag. It binds the baseline and candidate recipes, the
// admission evidence that earned the candidate its cohort, the cohort key
// domain and registered hash version that make assignment deterministic,
// the evidence-derived cohort share, the observation contract, and the
// typed rollback authority empowered to abort. Every bound is either an
// exact artifact identity or derived from one.
type RolloutPlan struct {
	Version        uint16             `json:"version"`
	Baseline       artifact.ID        `json:"baseline"`
	Candidate      artifact.ID        `json:"candidate"`
	Admission      artifact.ID        `json:"admission"`
	CohortKey      string             `json:"cohort_key"`
	HashVersion    uint16             `json:"hash_version"`
	CohortShare    uint32             `json:"cohort_share"`
	CohortEvidence artifact.ID        `json:"cohort_evidence"`
	Observation    RolloutObservation `json:"observation"`
	Rollback       artifact.ID        `json:"rollback"`
	ID             artifact.ID        `json:"-"`
}

var rolloutPlanCodec = artifact.JSONDocumentCodec(
	"rollout plan", artifact.KindProfile, RolloutPlanMediaType, RolloutPlanSchema,
	canonicalizeRolloutPlan,
	func(value RolloutPlan) artifact.ID { return value.ID },
	func(value *RolloutPlan, id artifact.ID) { value.ID = id }, nil,
)

func canonicalizeRolloutPlan(value *RolloutPlan) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid rollout plan version")
	}
	if value.Baseline.Kind() != artifact.KindRecipe || value.Candidate.Kind() != artifact.KindRecipe {
		return errors.New("run record: rollout requires exact baseline and candidate recipe identities")
	}
	if value.Baseline == value.Candidate {
		return errors.New("run record: rollout candidate must differ from its baseline")
	}
	if value.Admission.Kind() != artifact.KindEvidence {
		return errors.New("run record: rollout requires the admission evidence that earned the candidate its cohort")
	}
	if value.CohortKey == "" || !textcheck.Bounded(value.CohortKey, rolloutTextBytes, "\x00") {
		return errors.New("run record: rollout requires a bounded cohort key domain")
	}
	if value.HashVersion != RolloutHashVersion {
		return fmt.Errorf("run record: unregistered rollout hash version %d", value.HashVersion)
	}
	if value.CohortShare == 0 || value.CohortShare > RolloutCohortDenominator {
		return errors.New("run record: rollout cohort share must be a positive basis-point bound")
	}
	if value.CohortEvidence.Kind() != artifact.KindEvidence {
		return errors.New("run record: rollout cohort share must derive from exact evidence")
	}
	if value.Observation.Metric == "" || !textcheck.Bounded(value.Observation.Metric, rolloutTextBytes, "\x00") ||
		value.Observation.MinObservations == 0 || value.Observation.Evidence.Kind() != artifact.KindEvidence {
		return errors.New("run record: rollout requires an evidence-derived observation contract")
	}
	if value.Rollback.Kind() != artifact.KindEvidence {
		return errors.New("run record: rollout requires a typed rollback authority")
	}
	return nil
}

// NewRolloutPlan canonicalizes and identifies one immutable rollout plan.
func NewRolloutPlan(value RolloutPlan) (RolloutPlan, error) {
	return rolloutPlanCodec.NewPrepared(value, prepareRolloutPlan)
}

func prepareRolloutPlan(value *RolloutPlan) {
	value.Version = artifact.InitialDocumentVersion
	value.ID = artifact.ID{}
}

// ParseRolloutPlan decodes one canonical rollout plan document and proves
// its content identity.
func ParseRolloutPlan(content []byte) (RolloutPlan, error) {
	return rolloutPlanCodec.Parse(content)
}

// Lineage binds the plan to every exact authority it cites, each exactly
// once even when one evidence document derives several bounds.
func (value RolloutPlan) Lineage() []artifact.Lineage {
	return artifact.UniqueDependencyLineage(value.ID,
		value.Baseline, value.Candidate, value.Admission,
		value.CohortEvidence, value.Observation.Evidence, value.Rollback,
	)
}

// Batch wraps the plan as one committable store batch.
func (value RolloutPlan) Batch(key string) (artifact.Batch, error) {
	return rolloutPlanCodec.Batch(key, value, value.Lineage(), nil)
}

// Assign returns which arm serves one workload unit under this plan. The
// assignment is the registered version-1 scheme: a SHA-256 bucket over the
// plan's content identity, its cohort key domain, and the unit, reduced to
// basis points. It is a pure function of immutable inputs, so the same unit
// keeps the same arm across retry, replay, peer placement, and restart —
// no process state, ordering, or randomness participates.
func (value RolloutPlan) Assign(unit string) (artifact.ID, error) {
	if rolloutPlanCodec.ValidateIdentity(value) != nil {
		return artifact.ID{}, errors.New("run record: assignment requires an identified rollout plan")
	}
	if unit == "" || !textcheck.Bounded(unit, rolloutTextBytes, "\x00") {
		return artifact.ID{}, errors.New("run record: assignment requires a bounded workload unit")
	}
	digest := sha256.Sum256([]byte(value.ID.String() + "\x00" + value.CohortKey + "\x00" + unit))
	bucket := binary.BigEndian.Uint64(digest[:8]) % RolloutCohortDenominator
	if bucket < uint64(value.CohortShare) {
		return value.Candidate, nil
	}
	return value.Baseline, nil
}
