package runrecord

import (
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/textcheck"
)

const (
	ServingObservationVersion   uint16 = 1
	ServingObservationMediaType        = "application/vnd.overgo.serving-observation+json"
	ServingObservationSchema           = "overgo/serving-observation/v1"
)

var servingObservationCodec = artifact.JSONDocumentCodec(
	"serving observation", artifact.KindEvidence, ServingObservationMediaType, ServingObservationSchema,
	canonicalizeServingObservation,
	func(value ServingObservation) artifact.ID { return value.ID },
	func(value *ServingObservation, id artifact.ID) { value.ID = id },
	func(value ServingObservation) ServingObservation {
		value.Phases = slices.Clone(value.Phases)
		return value
	},
)

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
	Version       uint16           `json:"version"`
	Model         artifact.ID      `json:"model"`
	Recipe        artifact.ID      `json:"recipe"`
	Environment   artifact.ID      `json:"environment"`
	Operation     artifact.ID      `json:"operation,omitzero"`
	Task          recipe.Task      `json:"task"`
	Outcome       Outcome          `json:"outcome"`
	StartedUnixNS int64            `json:"started_unix_ns"`
	MeasuredNS    uint64           `json:"measured_ns"`
	SessionReused bool             `json:"session_reused,omitempty"`
	Usage         ServingUsage     `json:"usage"`
	Resources     ServingResources `json:"resources"`
	Phases        []PhaseMetric    `json:"phases,omitempty"`
	Failure       string           `json:"failure,omitempty"`
	ID            artifact.ID      `json:"-"`
}

func (value ServingObservation) ValidateIdentity() error {
	return servingObservationCodec.ValidateIdentity(value)
}

func (value ServingObservation) Content() (artifact.Content, error) {
	return servingObservationCodec.Content(value)
}

func (value ServingObservation) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Model, value.Recipe, value.Environment}
	if value.Operation.Valid() {
		parents = append(parents, value.Operation)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func (value ServingObservation) Batch(key string) (artifact.Batch, error) {
	return servingObservationCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeServingObservation(value *ServingObservation) error {
	if value == nil || value.Version != ServingObservationVersion || value.Model.Kind() != artifact.KindModel ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Environment.Kind() != artifact.KindEvidence ||
		!value.Task.Valid() || value.StartedUnixNS <= 0 || value.MeasuredNS == 0 || value.MeasuredNS > math.MaxInt64 {
		return errors.New("run record: invalid serving observation authority")
	}
	if value.Operation.Valid() && (value.Operation.Kind() != artifact.KindEvidence || value.Operation == value.Environment) {
		return errors.New("run record: invalid serving operation")
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
	return canonicalizePhases(&value.Phases)
}
