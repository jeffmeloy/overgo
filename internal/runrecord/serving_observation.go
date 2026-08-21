package runrecord

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	ServingObservationMediaType = "application/vnd.overgo.serving-observation+json"
	ServingObservationSchema    = "overgo/serving-observation/v1"
)

var servingObservationCodec = artifact.JSONDocumentCodec(
	"serving observation", artifact.KindEvidence, ServingObservationMediaType, ServingObservationSchema,
	canonicalizeServingObservation,
	func(value ServingObservation) artifact.ID { return value.ID },
	func(value *ServingObservation, id artifact.ID) { value.ID = id },
	func(value ServingObservation) ServingObservation {
		value.Phases = slices.Clone(value.Phases)
		value.Hardware = slices.Clone(value.Hardware)
		return value
	},
)

type ServingHardwareStage string

const (
	ServingHardwareStart   ServingHardwareStage = "start"
	ServingHardwarePrefill ServingHardwareStage = "prefill"
	ServingHardwareFinish  ServingHardwareStage = "finish"
)

// ServingHardwareSample: lifecycle-bound device allocation state.
type ServingHardwareSample struct {
	Stage              ServingHardwareStage `json:"stage"`
	ElapsedNS          uint64               `json:"elapsed_ns"`
	DeviceCurrentBytes uint64               `json:"device_current_bytes"`
	DevicePeakBytes    uint64               `json:"device_peak_bytes"`
	DeviceAllocations  uint64               `json:"device_allocations"`
}

// ServingUsage: payload-free request accounting.
type ServingUsage struct {
	InputTokens  uint64 `json:"input_tokens,omitempty"`
	OutputTokens uint64 `json:"output_tokens,omitempty"`
	InputBytes   uint64 `json:"input_bytes,omitempty"`
	OutputBytes  uint64 `json:"output_bytes,omitempty"`
}

// ServingResources: observed memory and transfer facts.
type ServingResources struct {
	PeakHostBytes     uint64 `json:"peak_host_bytes,omitempty"`
	PeakDeviceBytes   uint64 `json:"peak_device_bytes,omitempty"`
	HostToDeviceBytes uint64 `json:"host_to_device_bytes,omitempty"`
	DeviceToHostBytes uint64 `json:"device_to_host_bytes,omitempty"`
}

// ServingObservation: one recipe-bound serving attempt.
type ServingObservation struct {
	Version       uint16                  `json:"version"`
	Model         artifact.ID             `json:"model"`
	Recipe        artifact.ID             `json:"recipe"`
	Environment   artifact.ID             `json:"environment"`
	Operation     artifact.ID             `json:"operation,omitzero"`
	Run           artifact.ID             `json:"run,omitzero"`
	Task          recipe.Task             `json:"task"`
	Outcome       Outcome                 `json:"outcome"`
	StartedUnixNS int64                   `json:"started_unix_ns"`
	MeasuredNS    uint64                  `json:"measured_ns"`
	SessionReused bool                    `json:"session_reused,omitempty"`
	Usage         ServingUsage            `json:"usage"`
	Resources     ServingResources        `json:"resources"`
	Phases        []PhaseMetric           `json:"phases,omitempty"`
	Hardware      []ServingHardwareSample `json:"hardware,omitempty"`
	Failure       string                  `json:"failure,omitempty"`
	ID            artifact.ID             `json:"-"`
}

func (value ServingObservation) ValidateIdentity() error {
	return servingObservationCodec.ValidateIdentity(value)
}

func ParseServingObservation(content []byte) (ServingObservation, error) {
	return servingObservationCodec.Parse(content)
}

// RequireServingObservation: validated repository read.
func RequireServingObservation(ctx context.Context, reader artifact.Reader, id artifact.ID) (ServingObservation, error) {
	return servingObservationCodec.Require(ctx, reader, id)
}

func (value ServingObservation) Content() (artifact.Content, error) {
	return servingObservationCodec.Content(value)
}

func (value ServingObservation) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Model, value.Recipe, value.Environment}
	if value.Run.Valid() {
		parents = append(parents, value.Run)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func (value ServingObservation) Batch(key string) (artifact.Batch, error) {
	return servingObservationCodec.Batch(key, value, value.Lineage(), nil)
}

// PublishServingObservation: identify and commit one serving fact.
func PublishServingObservation(ctx context.Context, repository artifact.Repository, value ServingObservation) (ServingObservation, error) {
	if ctx == nil || repository == nil {
		return ServingObservation{}, errors.New("run record: serving observation repository is absent")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	identified, err := servingObservationCodec.New(value)
	if err != nil {
		return ServingObservation{}, err
	}
	batch, err := identified.Batch("serving/observation/" + identified.ID.String())
	if err != nil {
		return ServingObservation{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return ServingObservation{}, err
	}
	return identified, nil
}

func canonicalizeServingObservation(value *ServingObservation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Model.Kind() != artifact.KindModel ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Environment.Kind() != artifact.KindEvidence ||
		!value.Task.Valid() || value.StartedUnixNS <= 0 || value.MeasuredNS > math.MaxInt64 {
		return errors.New("run record: invalid serving observation authority")
	}
	if value.Operation.Valid() && (value.Operation.Kind() != artifact.KindEvidence || value.Operation == value.Environment) {
		return errors.New("run record: invalid serving operation")
	}
	if value.Run.Valid() && value.Run.Kind() != artifact.KindRun {
		return errors.New("run record: invalid serving run")
	}
	switch value.Outcome {
	case OutcomeSucceeded:
		if value.Failure != "" {
			return errors.New("run record: successful serving observation has failure")
		}
	case OutcomeFailed:
		if !textcheck.LowerIdentifier(value.Failure, maxLabelBytes) {
			return errors.New("run record: failed serving observation needs failure code")
		}
	case OutcomeCancelled:
		if value.Failure != "" {
			return errors.New("run record: cancelled serving observation has failure")
		}
	default:
		return errors.New("run record: invalid serving outcome")
	}
	if err := canonicalizePhases(&value.Phases); err != nil {
		return err
	}
	var elapsed uint64
	order := 0
	for index, sample := range value.Hardware {
		next := servingHardwareStageOrder(sample.Stage)
		if next <= order ||
			sample.DeviceCurrentBytes > sample.DevicePeakBytes || sample.ElapsedNS > value.MeasuredNS ||
			index > 0 && sample.ElapsedNS < elapsed {
			return errors.New("run record: invalid serving hardware sample")
		}
		order, elapsed = next, sample.ElapsedNS
	}
	return nil
}

func servingHardwareStageOrder(stage ServingHardwareStage) int {
	switch stage {
	case ServingHardwareStart:
		return 1
	case ServingHardwarePrefill:
		return 2
	case ServingHardwareFinish:
		return 3
	default:
		return 0
	}
}
