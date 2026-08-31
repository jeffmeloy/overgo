package runrecord

import (
	"context"
	"errors"
	"math"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	ServingObservationMediaType = "application/vnd.overgo.serving-observation+json"
	ServingObservationSchema    = "overgo/serving-observation/v1"
	// ServingAttemptAliasRoot scopes current serving attempts.
	ServingAttemptAliasRoot = "serving/attempt/"
)

var servingObservationCodec = artifact.JSONDocumentCodec(
	"serving observation", artifact.KindEvidence, ServingObservationMediaType, ServingObservationSchema,
	canonicalizeServingObservation,
	func(value ServingObservation) artifact.ID { return value.ID },
	func(value *ServingObservation, id artifact.ID) { value.ID = id },
	func(value ServingObservation) ServingObservation {
		value.Phases = slices.Clone(value.Phases)
		value.Hardware = slices.Clone(value.Hardware)
		value.Causal = cloneCausal(value.Causal)
		return value
	},
)

type ServingHardwareStage string

const (
	ServingHardwareStart   ServingHardwareStage = "start"
	ServingHardwarePrefill ServingHardwareStage = "prefill"
	ServingHardwareFinish  ServingHardwareStage = "finish"
)

// ServingHardwareSample defines lifecycle-bound device allocation state.
type ServingHardwareSample struct {
	Stage              ServingHardwareStage `json:"stage"`
	ElapsedNS          uint64               `json:"elapsed_ns"`
	DeviceCurrentBytes uint64               `json:"device_current_bytes"`
	DevicePeakBytes    uint64               `json:"device_peak_bytes"`
	DeviceAllocations  uint64               `json:"device_allocations"`
}

// ServingUsage defines payload-free request accounting.
type ServingUsage struct {
	InputTokens  uint64 `json:"input_tokens,omitempty"`
	OutputTokens uint64 `json:"output_tokens,omitempty"`
	InputBytes   uint64 `json:"input_bytes,omitempty"`
	OutputBytes  uint64 `json:"output_bytes,omitempty"`
}

// ServingResources defines observed memory and transfer facts.
type ServingResources struct {
	PeakHostBytes     uint64 `json:"peak_host_bytes,omitempty"`
	PeakDeviceBytes   uint64 `json:"peak_device_bytes,omitempty"`
	HostToDeviceBytes uint64 `json:"host_to_device_bytes,omitempty"`
	DeviceToHostBytes uint64 `json:"device_to_host_bytes,omitempty"`
}

// ServingAttemptKind classifies a validated attempt transition.
type ServingAttemptKind string

const (
	// ServingAttemptPrimary identifies the initial local execution.
	ServingAttemptPrimary ServingAttemptKind = "primary"
	// ServingAttemptRetry identifies repeated model and recipe execution.
	ServingAttemptRetry ServingAttemptKind = "retry"
	// ServingAttemptReselection identifies changed model or recipe execution.
	ServingAttemptReselection ServingAttemptKind = "reselection"
	// ServingAttemptSpillover identifies compatibility-bound peer execution.
	ServingAttemptSpillover ServingAttemptKind = "spillover"
)

// ServingObservation defines one recipe-bound serving attempt.
type ServingObservation struct {
	Version       uint16                  `json:"version"`
	Model         artifact.ID             `json:"model"`
	Recipe        artifact.ID             `json:"recipe"`
	Environment   artifact.ID             `json:"environment"`
	Operation     artifact.ID             `json:"operation,omitzero"`
	Run           artifact.ID             `json:"run,omitzero"`
	Previous      artifact.ID             `json:"previous,omitzero"`
	Compatibility artifact.ID             `json:"compatibility,omitzero"`
	Attempt       uint32                  `json:"attempt,omitempty"`
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
	// Causal explains why the serving execution occurred.
	Causal *CausalContext `json:"causal,omitempty"`
	ID     artifact.ID    `json:"-"`
}

func (value ServingObservation) ValidateIdentity() error {
	return servingObservationCodec.ValidateIdentity(value)
}

func ParseServingObservation(content []byte) (ServingObservation, error) {
	return servingObservationCodec.Parse(content)
}

// RequireServingObservation returns a validated repository record.
func RequireServingObservation(ctx context.Context, reader artifact.Reader, id artifact.ID) (ServingObservation, error) {
	return servingObservationCodec.Require(ctx, reader, id)
}

// RequireServingAttemptObservation loads one serving fact and proves that its
// operation and attempt ordinal select that exact immutable fact. Callers that
// only need to inspect historical content may continue to use
// RequireServingObservation.
func RequireServingAttemptObservation(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (ServingObservation, error) {
	observation, err := RequireServingObservation(ctx, reader, id)
	if err != nil {
		return ServingObservation{}, err
	}
	if !observation.Operation.Valid() {
		return ServingObservation{}, errors.New("run record: serving observation has no operation binding")
	}
	current, found, err := artifact.ResolveAlias(
		ctx, reader, servingAttemptAlias(observation.Operation, observation.Attempt),
	)
	if err != nil {
		return ServingObservation{}, err
	}
	if !found || current != observation.ID {
		return ServingObservation{}, errors.New("run record: serving observation differs from the operation attempt authority")
	}
	return observation, nil
}

func (value ServingObservation) Content() (artifact.Content, error) {
	return servingObservationCodec.Content(value)
}

func (value ServingObservation) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Model, value.Recipe, value.Environment}
	for _, parent := range []artifact.ID{value.Previous, value.Compatibility} {
		if parent.Valid() {
			parents = append(parents, parent)
		}
	}
	if value.Run.Valid() {
		parents = append(parents, value.Run)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

func (value ServingObservation) Batch(key string) (artifact.Batch, error) {
	alias := ServingAttemptAliasRoot + value.ID.String()
	if value.Operation.Valid() {
		alias = servingAttemptAlias(value.Operation, value.Attempt)
	}
	batch, err := servingObservationCodec.Batch(key, value, value.Lineage(), []artifact.AliasBinding{{Name: alias, Target: value.ID}})
	if err != nil {
		return artifact.Batch{}, err
	}
	if err := BindCausality(&batch, value.ID, value.Causal); err != nil {
		return artifact.Batch{}, err
	}
	return batch, nil
}

// NewServingObservation validates and identifies one immutable serving fact
// without committing it, allowing callers to publish a larger atomic graph.
func NewServingObservation(value ServingObservation) (ServingObservation, error) {
	return servingObservationCodec.NewInitial(value)
}

// PublishServingObservation identifies and commits one serving fact.
func PublishServingObservation(ctx context.Context, repository artifact.Repository, value ServingObservation) (ServingObservation, error) {
	if ctx == nil || repository == nil {
		return ServingObservation{}, errors.New("run record: serving observation repository is absent")
	}
	identified, err := NewServingObservation(value)
	if err != nil {
		return ServingObservation{}, err
	}
	if err := validateServingAttempt(ctx, repository, identified); err != nil {
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

// AttemptKind classifies the effective route transition.
func (value ServingObservation) AttemptKind(previous *ServingObservation) ServingAttemptKind {
	if value.Compatibility.Valid() {
		return ServingAttemptSpillover
	}
	if previous == nil {
		return ServingAttemptPrimary
	}
	if value.Model == previous.Model && value.Recipe == previous.Recipe {
		return ServingAttemptRetry
	}
	return ServingAttemptReselection
}

// ResourceFitness converts the serving record's explicitly observed scalar
// fields into the common resource contract. observed is required because the
// legacy serving schema used zero for both unknown and measured zero; callers
// must name exactly which fields their instrumentation measured.
func (value ServingObservation) ResourceFitness(
	provider artifact.ID,
	interactions *InteractionWork,
	observed ...ResourceMetric,
) (ResourceFitness, error) {
	if err := value.ValidateIdentity(); err != nil {
		return ResourceFitness{}, err
	}
	measures := make([]ResourceMeasure, 0, len(observed))
	for _, metric := range observed {
		var measured uint64
		switch metric {
		case ResourceInputTokens:
			measured = value.Usage.InputTokens
		case ResourceOutputTokens:
			measured = value.Usage.OutputTokens
		case ResourceInputBytes:
			measured = value.Usage.InputBytes
		case ResourceOutputBytes:
			measured = value.Usage.OutputBytes
		case ResourceWallNS:
			measured = value.MeasuredNS
		case ResourcePeakHostBytes:
			measured = value.Resources.PeakHostBytes
		case ResourcePeakDeviceBytes:
			measured = value.Resources.PeakDeviceBytes
		case ResourceHostToDeviceBytes:
			measured = value.Resources.HostToDeviceBytes
		case ResourceDeviceToHostBytes:
			measured = value.Resources.DeviceToHostBytes
		default:
			return ResourceFitness{}, errors.New("run record: serving record does not own requested resource metric")
		}
		measures = append(measures, ResourceMeasure{Metric: metric, Value: measured})
	}
	return NewResourceFitness(ResourceFitness{
		Scope: ResourceScope{
			Surface: SurfaceServing, Model: value.Model, Hardware: value.Environment,
			Provider: provider, Workload: value.Recipe, Attempt: value.ID,
		},
		Measures: measures, Interactions: interactions,
	})
}

func validateServingAttempt(ctx context.Context, reader artifact.Reader, value ServingObservation) error {
	if !value.Previous.Valid() {
		return nil
	}
	previous, err := RequireServingObservation(ctx, reader, value.Previous)
	if err != nil {
		return err
	}
	previousAttempt := value.Attempt
	previousAttempt--
	previousID, found, err := artifact.ResolveAlias(ctx, reader, servingAttemptAlias(value.Operation, previousAttempt))
	if err != nil || !found || previousID != value.Previous {
		return errors.Join(errors.New("run record: previous serving attempt differs"), err)
	}
	nextAttempt := previous.Attempt
	nextAttempt++
	if previous.Outcome != OutcomeFailed || value.Attempt != nextAttempt ||
		value.Operation != previous.Operation || value.Task != previous.Task ||
		value.Environment != previous.Environment {
		return errors.New("run record: serving attempt chain differs")
	}
	return nil
}

func servingAttemptAlias(operation artifact.ID, attempt uint32) string {
	return indexedAlias(ServingAttemptAliasRoot, operation, uint64(attempt))
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
	if value.Previous.Valid() && (value.Previous.Kind() != artifact.KindEvidence || !value.Operation.Valid() ||
		value.Previous == value.Operation || value.Previous == value.Environment || value.Attempt == 0) ||
		!value.Previous.Valid() && value.Attempt != 0 {
		return errors.New("run record: invalid serving attempt")
	}
	if value.Compatibility.Valid() && (value.Compatibility.Kind() != artifact.KindEvidence ||
		value.Compatibility == value.Operation || value.Compatibility == value.Environment || value.Compatibility == value.Previous) {
		return errors.New("run record: invalid serving compatibility")
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
		if !validLabel(value.Failure) {
			return errors.New("run record: failed serving observation needs failure code")
		}
	case OutcomeCancelled:
		if value.Failure != "" {
			return errors.New("run record: cancelled serving observation has failure")
		}
	default:
		return errors.New("run record: invalid serving outcome")
	}
	if err := errors.Join(canonicalizePhases(&value.Phases), validCausal(value.Causal)); err != nil {
		return err
	}
	var elapsed uint64
	var previousStage ServingHardwareStage
	for index, sample := range value.Hardware {
		if !servingHardwareStageFollows(previousStage, sample.Stage) ||
			sample.DeviceCurrentBytes > sample.DevicePeakBytes || sample.ElapsedNS > value.MeasuredNS ||
			index > 0 && sample.ElapsedNS < elapsed {
			return errors.New("run record: invalid serving hardware sample")
		}
		previousStage, elapsed = sample.Stage, sample.ElapsedNS
	}
	return nil
}

func servingHardwareStageFollows(previous, next ServingHardwareStage) bool {
	switch previous {
	case "":
		return next == ServingHardwareStart || next == ServingHardwarePrefill || next == ServingHardwareFinish
	case ServingHardwareStart:
		return next == ServingHardwarePrefill || next == ServingHardwareFinish
	case ServingHardwarePrefill:
		return next == ServingHardwareFinish
	default:
		return false
	}
}
