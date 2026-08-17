package runrecord

import (
	"errors"
	"fmt"
	"slices"
	"time"

	"overgo/internal/artifact"
)

const (
	ExperimentLifecycleVersion   uint16 = 1
	ExperimentLifecycleMediaType        = "application/vnd.overgo.experiment-lifecycle+json"
	ExperimentLifecycleSchema           = "overgo/experiment-lifecycle/v1"
)

// ExperimentState is one station of the runtime experiment state machine.
type ExperimentState string

const (
	ExperimentProposed   ExperimentState = "proposed"
	ExperimentAdmitted   ExperimentState = "admitted"
	ExperimentLeased     ExperimentState = "leased"
	ExperimentRunning    ExperimentState = "running"
	ExperimentEvaluated  ExperimentState = "evaluated"
	ExperimentRefused    ExperimentState = "refused"
	ExperimentPromoted   ExperimentState = "promoted"
	ExperimentContained  ExperimentState = "contained"
	ExperimentRolledBack ExperimentState = "rolled-back"
)

// experimentTransitions is the legal forward edge set. Recovery -- re-leasing
// an expired lease or a dead runner -- is the retry-incrementing edge back to
// leased; everything else advances or contains.
var experimentTransitions = map[ExperimentState][]ExperimentState{
	ExperimentProposed:   {ExperimentAdmitted},
	ExperimentAdmitted:   {ExperimentLeased, ExperimentContained},
	ExperimentLeased:     {ExperimentRunning, ExperimentLeased, ExperimentContained},
	ExperimentRunning:    {ExperimentEvaluated, ExperimentLeased, ExperimentContained},
	ExperimentEvaluated:  {ExperimentRefused, ExperimentPromoted},
	ExperimentRefused:    {},
	ExperimentPromoted:   {ExperimentRolledBack},
	ExperimentContained:  {ExperimentRolledBack},
	ExperimentRolledBack: {},
}

// ExperimentLifecycle is one immutable state-transition record. The chain of
// records linked by Prior is the experiment's whole runtime history; identical
// transition facts produce the identical content-addressed ID, which is what
// makes replay idempotent -- re-recording a transition is a no-op by identity,
// never a duplicate.
type ExperimentLifecycle struct {
	Version    uint16          `json:"version"`
	State      ExperimentState `json:"state"`
	Experiment artifact.ID     `json:"experiment"`
	// Retry is the recovery identity: it increments exactly when an expired
	// lease or dead runner is re-leased, so retried work is attributable and
	// never silently conflated with its predecessor.
	Retry uint32       `json:"retry"`
	Prior *artifact.ID `json:"prior,omitempty"`
	// Evidence binds the transition to the immutable fact justifying it: the
	// admission decision, the work lease, the run record, the verdict.
	Evidence artifact.ID `json:"evidence"`
	// HeartbeatExpiry bounds leased and running states; past the bound the
	// record is recoverable, never silently live.
	HeartbeatExpiry string `json:"heartbeat_expiry,omitempty"`
	// Checkpoint is the recovery point a retried run resumes from.
	Checkpoint *artifact.ID `json:"checkpoint,omitempty"`
	ID         artifact.ID  `json:"-"`
}

var experimentLifecycleCodec = artifact.JSONDocumentCodec(
	"experiment lifecycle", artifact.KindEvidence, ExperimentLifecycleMediaType, ExperimentLifecycleSchema,
	canonicalizeExperimentLifecycle,
	func(value ExperimentLifecycle) artifact.ID { return value.ID },
	func(value *ExperimentLifecycle, id artifact.ID) { value.ID = id }, nil,
)

// NewExperimentLifecycle identifies one transition. prior is nil exactly for
// the initial proposed record; every other transition must name its
// predecessor and be a legal edge of the state machine.
func NewExperimentLifecycle(value ExperimentLifecycle, prior *ExperimentLifecycle) (ExperimentLifecycle, error) {
	value.Version = ExperimentLifecycleVersion
	value.ID = artifact.ID{}
	if prior == nil {
		if value.State != ExperimentProposed || value.Retry != 0 || value.Prior != nil {
			return ExperimentLifecycle{}, errors.New("run record: experiment history begins proposed at retry 0")
		}
		return experimentLifecycleCodec.New(value)
	}
	if !prior.ID.Valid() {
		return ExperimentLifecycle{}, errors.New("run record: prior experiment record lacks identity")
	}
	if value.Experiment != prior.Experiment {
		return ExperimentLifecycle{}, errors.New("run record: transition changes experiment identity")
	}
	if !slices.Contains(experimentTransitions[prior.State], value.State) {
		return ExperimentLifecycle{}, fmt.Errorf("run record: illegal experiment transition %s -> %s", prior.State, value.State)
	}
	recovery := value.State == ExperimentLeased && (prior.State == ExperimentLeased || prior.State == ExperimentRunning)
	if recovery {
		if value.Retry != prior.Retry+1 {
			return ExperimentLifecycle{}, errors.New("run record: recovery must increment the retry identity")
		}
	} else if value.Retry != prior.Retry {
		return ExperimentLifecycle{}, errors.New("run record: retry identity changes only on recovery")
	}
	priorID := prior.ID
	value.Prior = &priorID
	return experimentLifecycleCodec.New(value)
}

func ParseExperimentLifecycle(content []byte) (ExperimentLifecycle, error) {
	return experimentLifecycleCodec.Parse(content)
}

func (value ExperimentLifecycle) Content() (artifact.Content, error) {
	return experimentLifecycleCodec.Content(value)
}

func canonicalizeExperimentLifecycle(value *ExperimentLifecycle) error {
	if value == nil || value.Version != ExperimentLifecycleVersion {
		return errors.New("run record: invalid experiment lifecycle version")
	}
	if _, known := experimentTransitions[value.State]; !known {
		return fmt.Errorf("run record: invalid experiment state %q", value.State)
	}
	if value.Experiment.Kind() != artifact.KindEvidence {
		return errors.New("run record: experiment identity must be an evidence artifact")
	}
	if !value.Evidence.Valid() {
		return errors.New("run record: experiment transition lacks justifying evidence")
	}
	if (value.State == ExperimentProposed) != (value.Prior == nil) {
		return errors.New("run record: exactly the proposed record has no prior")
	}
	if value.Prior != nil && value.Prior.Kind() != artifact.KindEvidence {
		return errors.New("run record: prior reference must be an evidence artifact")
	}
	bounded := value.State == ExperimentLeased || value.State == ExperimentRunning
	if bounded {
		if _, err := time.Parse(time.RFC3339Nano, value.HeartbeatExpiry); err != nil {
			return errors.New("run record: leased and running states require a heartbeat expiry")
		}
	} else if value.HeartbeatExpiry != "" {
		return errors.New("run record: heartbeat expiry outside leased or running")
	}
	if value.Checkpoint != nil && value.Checkpoint.Kind() != artifact.KindCheckpoint {
		return errors.New("run record: experiment checkpoint reference kind mismatch")
	}
	return nil
}

// ExperimentStatus is the reconciled truth for one experiment: the chain tip,
// whether its liveness bound has lapsed, and the retry identity a recovery
// must use.
type ExperimentStatus struct {
	Tip       ExperimentLifecycle
	Expired   bool
	NextRetry uint32
}

// ReconcileExperiment folds one experiment's immutable records back into its
// current state. Duplicate records collapse by identity; a diverged chain is
// an error, never a guess. An expired tip is reported recoverable with the
// incremented retry -- recovery is an explicit recorded transition, so crash
// and replay converge on the same chain.
func ReconcileExperiment(records []ExperimentLifecycle, now time.Time) (ExperimentStatus, error) {
	if len(records) == 0 {
		return ExperimentStatus{}, errors.New("run record: no experiment records to reconcile")
	}
	byID := map[artifact.ID]ExperimentLifecycle{}
	experiment := records[0].Experiment
	for _, record := range records {
		if !record.ID.Valid() {
			return ExperimentStatus{}, errors.New("run record: unidentified experiment record")
		}
		if record.Experiment != experiment {
			return ExperimentStatus{}, errors.New("run record: records span multiple experiments")
		}
		byID[record.ID] = record
	}
	referenced := map[artifact.ID]bool{}
	roots := 0
	for _, record := range byID {
		if record.Prior == nil {
			roots++
			continue
		}
		prior, ok := byID[*record.Prior]
		if !ok {
			return ExperimentStatus{}, errors.New("run record: experiment chain references an absent prior")
		}
		if !slices.Contains(experimentTransitions[prior.State], record.State) {
			return ExperimentStatus{}, fmt.Errorf("run record: recorded chain holds illegal transition %s -> %s", prior.State, record.State)
		}
		referenced[*record.Prior] = true
	}
	if roots != 1 {
		return ExperimentStatus{}, fmt.Errorf("run record: experiment chain has %d roots, want 1", roots)
	}
	var tips []ExperimentLifecycle
	for id, record := range byID {
		if !referenced[id] {
			tips = append(tips, record)
		}
	}
	if len(tips) != 1 {
		return ExperimentStatus{}, fmt.Errorf("run record: experiment chain diverged into %d tips", len(tips))
	}
	status := ExperimentStatus{Tip: tips[0]}
	if status.Tip.State == ExperimentLeased || status.Tip.State == ExperimentRunning {
		expiry, err := time.Parse(time.RFC3339Nano, status.Tip.HeartbeatExpiry)
		if err != nil {
			return ExperimentStatus{}, err
		}
		if now.After(expiry) {
			status.Expired = true
			status.NextRetry = status.Tip.Retry + 1
		}
	}
	return status, nil
}
