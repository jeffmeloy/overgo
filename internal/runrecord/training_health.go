package runrecord

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	trainingHealthMediaType = "application/vnd.overgo.training-health+json"
	trainingHealthSchema    = "overgo/training-health/v1"
)

type trainingHealthPhase string

const (
	trainingHealthLoading    trainingHealthPhase = "loading"
	trainingHealthForward    trainingHealthPhase = "forward"
	trainingHealthBackward   trainingHealthPhase = "backward"
	trainingHealthOptimize   trainingHealthPhase = "optimize"
	trainingHealthCheckpoint trainingHealthPhase = "checkpoint"
)

var trainingHealthPhases = [...]trainingHealthPhase{
	trainingHealthLoading, trainingHealthForward, trainingHealthBackward, trainingHealthOptimize, trainingHealthCheckpoint,
}

type trainingHealthFloat struct {
	Observed bool    `json:"observed"`
	Value    float64 `json:"value"`
}

type trainingHealthThroughput struct {
	Observed bool   `json:"observed"`
	Tokens   uint64 `json:"tokens"`
	WallNS   uint64 `json:"wall_ns"`
}

type trainingHealthLoadingValue struct {
	Observed bool   `json:"observed"`
	Bytes    uint64 `json:"bytes"`
	WallNS   uint64 `json:"wall_ns"`
}

type trainingHealthObservation struct {
	Ordinal            uint64                     `json:"ordinal"`
	Step               uint64                     `json:"step"`
	Phase              trainingHealthPhase        `json:"phase"`
	Authority          artifact.ID                `json:"authority"`
	Loss               trainingHealthFloat        `json:"loss"`
	GradientL2         trainingHealthFloat        `json:"gradient_l2"`
	Throughput         trainingHealthThroughput   `json:"throughput"`
	Loading            trainingHealthLoadingValue `json:"loading"`
	Checkpoint         artifact.ID                `json:"checkpoint,omitzero"`
	CheckpointEvidence artifact.ID                `json:"checkpoint_evidence,omitzero"`
}

type trainingHealth struct {
	ID           artifact.ID                 `json:"-"`
	Version      uint16                      `json:"version"`
	Run          artifact.ID                 `json:"run"`
	Program      artifact.ID                 `json:"program"`
	Recipe       artifact.ID                 `json:"recipe"`
	Model        artifact.ID                 `json:"model"`
	Dataset      artifact.ID                 `json:"dataset"`
	Split        artifact.ID                 `json:"split"`
	Code         artifact.ID                 `json:"code"`
	Environment  artifact.ID                 `json:"environment"`
	Hardware     artifact.ID                 `json:"hardware"`
	Plan         artifact.ID                 `json:"plan"`
	PlannedSteps uint64                      `json:"planned_steps"`
	Observations []trainingHealthObservation `json:"observations"`
}

var trainingHealthCodec = artifact.JSONDocumentCodec(
	"training health", artifact.KindEvidence, trainingHealthMediaType, trainingHealthSchema,
	canonicalizeTrainingHealth,
	func(value trainingHealth) artifact.ID { return value.ID },
	func(value *trainingHealth, id artifact.ID) { value.ID = id },
	func(value trainingHealth) trainingHealth {
		value.Observations = slices.Clone(value.Observations)
		return value
	},
)

var trainingHealthLineage = func(value trainingHealth) []artifact.Lineage {
	parents := []artifact.ID{
		value.Run, value.Program, value.Recipe, value.Model, value.Dataset, value.Split,
		value.Code, value.Environment, value.Hardware, value.Plan,
	}
	for _, observation := range value.Observations {
		parents = append(parents, observation.Authority, observation.Checkpoint, observation.CheckpointEvidence)
	}
	valid := parents[:0]
	for _, parent := range parents {
		if parent.Valid() {
			valid = append(valid, parent)
		}
	}
	return artifact.DependencyLineage(value.ID, uniqueScalingAuthorities(valid)...)
}

func canonicalizeTrainingHealth(value *trainingHealth) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Run.Kind() != artifact.KindRun ||
		value.Program.Kind() != artifact.KindRecipe || value.Recipe.Kind() != artifact.KindRecipe ||
		value.Model.Kind() != artifact.KindModel || value.Dataset.Kind() != artifact.KindDataset ||
		value.Split.Kind() != artifact.KindDatasetShard || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence || value.Hardware.Kind() != artifact.KindEvidence ||
		value.Plan.Kind() != artifact.KindProfile || value.PlannedSteps == 0 || len(value.Observations) == 0 {
		return errors.New("run record: invalid training health authorities or plan")
	}
	value.Observations = slices.Clone(value.Observations)
	slices.SortFunc(value.Observations, func(left, right trainingHealthObservation) int {
		return cmp.Compare(left.Ordinal, right.Ordinal)
	})
	seenAuthorities := make(map[artifact.ID]bool, len(value.Observations))
	lossSeen, gradientSeen, throughputSeen, loadingSeen, checkpointSeen := false, false, false, false, false
	var priorStep uint64
	for index, observation := range value.Observations {
		if observation.Ordinal != uint64(index) || observation.Step > value.PlannedSteps || index > 0 && observation.Step < priorStep ||
			observation.Authority.Kind() != artifact.KindEvidence || seenAuthorities[observation.Authority] ||
			!slices.Contains(trainingHealthPhases[:], observation.Phase) || !validTrainingHealthFloat(observation.Loss, false) ||
			!validTrainingHealthFloat(observation.GradientL2, true) ||
			observation.Throughput.Observed && observation.Throughput.WallNS == 0 ||
			!observation.Throughput.Observed && (observation.Throughput.Tokens != 0 || observation.Throughput.WallNS != 0) ||
			observation.Loading.Observed && observation.Loading.WallNS == 0 ||
			!observation.Loading.Observed && (observation.Loading.Bytes != 0 || observation.Loading.WallNS != 0) ||
			observation.Checkpoint.Valid() != observation.CheckpointEvidence.Valid() ||
			observation.Checkpoint.Valid() && (observation.Checkpoint.Kind() != artifact.KindCheckpoint || observation.CheckpointEvidence.Kind() != artifact.KindEvidence) {
			return errors.New("run record: invalid training health observation")
		}
		seenAuthorities[observation.Authority] = true
		priorStep = observation.Step
		lossSeen = lossSeen || observation.Loss.Observed
		gradientSeen = gradientSeen || observation.GradientL2.Observed
		throughputSeen = throughputSeen || observation.Throughput.Observed
		loadingSeen = loadingSeen || observation.Loading.Observed
		checkpointSeen = checkpointSeen || observation.Checkpoint.Valid()
	}
	if !lossSeen || !gradientSeen || !throughputSeen || !loadingSeen || !checkpointSeen {
		return errors.New("run record: training health signal coverage is incomplete")
	}
	return nil
}

func validTrainingHealthFloat(value trainingHealthFloat, nonnegative bool) bool {
	if !value.Observed {
		return value.Value == 0
	}
	return checked.Finite64(value.Value) && (!nonnegative || value.Value >= 0)
}
