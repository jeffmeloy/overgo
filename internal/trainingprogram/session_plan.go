package trainingprogram

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/optimizer"
)

// SessionSpec supplies compiled model, data, optimizer, and run facts.
type SessionSpec struct {
	Objective        ObjectiveKind
	DatasetUnits     int
	RequestedUpdates int
	MaximumSequence  int
	ObjectiveScale   float64
	MaxProjectedWall time.Duration
	Optimizer        optimizer.Config
}

// ProbeSpec supplies bounded evidence-run facts.
type ProbeSpec struct {
	Objective        ObjectiveKind
	Updates          int
	MaximumSequence  int
	ObjectiveScale   float64
	MaxProjectedWall time.Duration
	Parameters       int
	Optimizer        OptimizerPolicy
}

// TrainingSessionPlan seals execution policy for one training run.
type TrainingSessionPlan struct {
	id               artifact.ID
	updates          int
	maximumSequence  int
	objectiveScale   float64
	maxProjectedWall time.Duration
	optimizer        optimizer.Config
}

// CompileTrainingSessionPlan resolves omitted updates from dataset facts.
func CompileTrainingSessionPlan(spec SessionSpec) (TrainingSessionPlan, error) {
	updates := spec.RequestedUpdates
	if updates == 0 {
		updates = spec.DatasetUnits
	}
	if !validObjectiveKind(spec.Objective) || spec.DatasetUnits < 0 ||
		(spec.DatasetUnits == 0 && spec.RequestedUpdates == 0) || updates <= 0 ||
		spec.MaximumSequence < 0 || spec.MaxProjectedWall < 0 {
		return TrainingSessionPlan{}, errors.New("training program: invalid session facts")
	}
	if spec.Objective == ObjectiveDPO || spec.Objective == ObjectiveGRPO {
		if spec.ObjectiveScale <= 0 || !finite(spec.ObjectiveScale) {
			return TrainingSessionPlan{}, errors.New("training program: RL session requires finite positive objective scale")
		}
	} else if spec.ObjectiveScale != 0 {
		return TrainingSessionPlan{}, errors.New("training program: non-RL session rejects objective scale")
	}
	if err := spec.Optimizer.Validate(); err != nil {
		return TrainingSessionPlan{}, err
	}
	spec.Optimizer.Steps = updates
	body := struct {
		Objective        ObjectiveKind    `json:"objective"`
		DatasetUnits     int              `json:"dataset_units"`
		Updates          int              `json:"updates"`
		MaximumSequence  int              `json:"maximum_sequence"`
		ObjectiveScale   float64          `json:"objective_scale,omitempty"`
		MaxProjectedWall int64            `json:"max_projected_wall_ns,omitempty"`
		Optimizer        optimizer.Config `json:"optimizer"`
	}{
		Objective: spec.Objective, DatasetUnits: spec.DatasetUnits, Updates: updates,
		MaximumSequence: spec.MaximumSequence, ObjectiveScale: spec.ObjectiveScale,
		MaxProjectedWall: spec.MaxProjectedWall.Nanoseconds(), Optimizer: spec.Optimizer,
	}
	id, err := artifact.JSONID(artifact.KindProfile, body)
	if err != nil {
		return TrainingSessionPlan{}, err
	}
	return TrainingSessionPlan{
		id: id, updates: updates, maximumSequence: spec.MaximumSequence,
		objectiveScale: spec.ObjectiveScale, maxProjectedWall: spec.MaxProjectedWall,
		optimizer: spec.Optimizer,
	}, nil
}

// CompileProbeSessionPlan seals one explicitly bounded evidence run.
func CompileProbeSessionPlan(spec ProbeSpec) (TrainingSessionPlan, error) {
	config, err := spec.Optimizer.Config(spec.Parameters)
	if err != nil {
		return TrainingSessionPlan{}, err
	}
	return CompileTrainingSessionPlan(SessionSpec{
		Objective: spec.Objective, RequestedUpdates: spec.Updates,
		MaximumSequence: spec.MaximumSequence, ObjectiveScale: spec.ObjectiveScale,
		MaxProjectedWall: spec.MaxProjectedWall, Optimizer: config,
	})
}

// AdmitStepWall checks first-step projection against the compiled bound.
func (plan TrainingSessionPlan) AdmitStepWall(measured time.Duration) error {
	if measured < 0 {
		return errors.New("training program: negative step wall")
	}
	if plan.maxProjectedWall > 0 && measured*time.Duration(plan.updates) > plan.maxProjectedWall {
		return errors.New("training program: projected wall exceeds session bound")
	}
	return nil
}

// Seed derives a stream-separated deterministic seed from plan identity.
func (plan TrainingSessionPlan) Seed(stream string) (int64, error) {
	if !plan.id.Valid() || stream == "" {
		return 0, errors.New("training program: seed authority is incomplete")
	}
	id, err := artifact.JSONID(artifact.KindProfile, struct {
		Plan   artifact.ID `json:"plan"`
		Stream string      `json:"stream"`
	}{Plan: plan.id, Stream: stream})
	if err != nil {
		return 0, err
	}
	digest, err := hex.DecodeString(id.DigestHex())
	if err != nil {
		return 0, fmt.Errorf("training program: derive seed: %w", err)
	}
	if len(digest) < binary.Size(uint64(plan.updates)) {
		return 0, errors.New("training program: seed digest is incomplete")
	}
	return int64(binary.LittleEndian.Uint64(digest) & math.MaxInt64), nil
}

// ID returns immutable plan identity.
func (plan TrainingSessionPlan) ID() artifact.ID { return plan.id }

// Updates returns admitted update count.
func (plan TrainingSessionPlan) Updates() int { return plan.updates }

// MaximumSequence returns admitted sequence extent.
func (plan TrainingSessionPlan) MaximumSequence() int { return plan.maximumSequence }

// ObjectiveScale returns compiled objective scaling.
func (plan TrainingSessionPlan) ObjectiveScale() float64 { return plan.objectiveScale }

// MaxProjectedWall returns admitted wall bound.
func (plan TrainingSessionPlan) MaxProjectedWall() time.Duration { return plan.maxProjectedWall }

// Optimizer returns compiled optimizer configuration.
func (plan TrainingSessionPlan) Optimizer() optimizer.Config { return plan.optimizer }
