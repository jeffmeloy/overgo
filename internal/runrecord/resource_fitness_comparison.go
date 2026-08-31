package runrecord

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// ResourceFitnessComparisonMediaType identifies an immutable multi-lane
	// resource comparison.
	ResourceFitnessComparisonMediaType = "application/vnd.overgo.resource-fitness-comparison+json"
	// ResourceFitnessComparisonSchema identifies the comparison wire contract.
	ResourceFitnessComparisonSchema = "overgo/resource-fitness-comparison/v1"

	// ResourceLaneAgent is the canonical agent-work lane token.
	ResourceLaneAgent = "agent"
	// ResourceLaneEvaluation is the canonical evaluation-work lane token.
	ResourceLaneEvaluation = "evaluation"
	// ResourceLaneTool is the canonical tool-work lane token.
	ResourceLaneTool = "tool"
)

// ResourceFitnessLane binds one comparable subsystem lane to exact baseline
// and candidate observation streams. RequiredMetrics is the minimum metric
// contract; both streams must additionally expose the same complete sample
// protocol and metric keysets, and every observed metric is compared.
type ResourceFitnessLane struct {
	Name                string            `json:"name"`
	RequiredMetrics     []ResourceMetric  `json:"required_metrics,omitempty"`
	RequireInteractions bool              `json:"require_interactions,omitzero"`
	Baseline            ObservationStream `json:"baseline"`
	Candidate           ObservationStream `json:"candidate"`
	StrictMetrics       []ResourceMeasure `json:"strict_metrics,omitempty"`
	StrictInteractions  *InteractionWork  `json:"strict_interactions,omitempty"`
}

// ResourceFitnessComparison is an immutable no-regression proof over every
// declared workload/surface lane. Construction refuses missing observations,
// incomparable metric keysets, or any candidate resource/interaction increase.
// StrictImprovement reports whether at least one compared dimension decreased.
type ResourceFitnessComparison struct {
	Version           uint16                `json:"version"`
	Lanes             []ResourceFitnessLane `json:"lanes"`
	StrictImprovement bool                  `json:"strict_improvement"`
	ID                artifact.ID           `json:"-"`
}

var resourceFitnessComparisonCodec = artifact.JSONDocumentCodec(
	"run record resource fitness comparison", artifact.KindEvidence,
	ResourceFitnessComparisonMediaType, ResourceFitnessComparisonSchema,
	canonicalizeResourceFitnessComparison,
	func(value ResourceFitnessComparison) artifact.ID { return value.ID },
	func(value *ResourceFitnessComparison, id artifact.ID) { value.ID = id },
	cloneResourceFitnessComparison,
)

// NewResourceFitnessComparison validates, orders, and identifies one
// structurally canonical multi-lane comparison. It does not have a Reader and
// therefore cannot prove that the embedded stream aggregates match their raw
// chunks. Producers should use CompareResourceFitness; consumers should use
// RequireResourceFitnessComparison.
func NewResourceFitnessComparison(lanes []ResourceFitnessLane) (ResourceFitnessComparison, error) {
	return resourceFitnessComparisonCodec.NewInitial(ResourceFitnessComparison{
		Lanes: cloneResourceFitnessLanes(lanes),
	})
}

// CompareResourceFitness constructs a comparison only after replaying every
// endpoint from its exact immutable summary head. Current attempt aliases are
// deliberately irrelevant.
func CompareResourceFitness(
	ctx context.Context,
	reader artifact.Reader,
	lanes []ResourceFitnessLane,
) (ResourceFitnessComparison, error) {
	value, err := NewResourceFitnessComparison(lanes)
	if err != nil {
		return ResourceFitnessComparison{}, err
	}
	if err := verifyResourceFitnessComparisonSources(ctx, reader, value); err != nil {
		return ResourceFitnessComparison{}, err
	}
	return value, nil
}

// RequireResourceFitnessComparison loads a comparison and replays both exact
// historical stream heads for every lane. It never relies on current aliases.
func RequireResourceFitnessComparison(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (ResourceFitnessComparison, error) {
	return resourceFitnessComparisonCodec.RequireVerified(ctx, reader, id, verifyResourceFitnessComparisonSources)
}

func verifyResourceFitnessComparisonSources(
	ctx context.Context,
	reader artifact.Reader,
	value ResourceFitnessComparison,
) error {
	if ctx == nil || reader == nil {
		return errors.New("run record: resource fitness comparison source reader is absent")
	}
	for _, lane := range value.Lanes {
		_, baselineCostObserved := lane.Baseline.Aggregate.Measure(ResourceCostUnits)
		_, candidateCostObserved := lane.Candidate.Aggregate.Measure(ResourceCostUnits)
		if slices.Contains(lane.RequiredMetrics, ResourceCostUnits) || baselineCostObserved || candidateCostObserved {
			provider := lane.Baseline.Scope.Provider
			if !provider.Valid() {
				return fmt.Errorf("run record: resource fitness lane %q cost provider is absent", lane.Name)
			}
			if _, err := RequireCapabilityIdentity(ctx, reader, provider); err != nil {
				return errors.Join(err, fmt.Errorf("run record: resource fitness lane %q cost provider is not an exact capability", lane.Name))
			}
		}
		for _, endpoint := range []struct {
			name   string
			stream ObservationStream
		}{
			{name: "baseline", stream: lane.Baseline},
			{name: "candidate", stream: lane.Candidate},
		} {
			head := endpoint.stream.SummaryIDs[len(endpoint.stream.SummaryIDs)-1]
			replayed, requireErr := RequireObservationStream(
				ctx, reader, endpoint.stream.Scope.Attempt, head,
				ObservationStreamBounds{
					MaxChunks:   len(endpoint.stream.SummaryIDs),
					MaxRawBytes: endpoint.stream.Coverage.RawBytes,
				},
			)
			if requireErr != nil || !reflect.DeepEqual(replayed, endpoint.stream) {
				return errors.Join(
					requireErr,
					fmt.Errorf("run record: resource fitness lane %q %s stream differs from exact sources", lane.Name, endpoint.name),
				)
			}
		}
	}
	return nil
}

// SourceChunks returns the sorted unique raw chunks cited by the comparison.
func (value ResourceFitnessComparison) SourceChunks() []artifact.ID {
	var chunks []artifact.ID
	for _, lane := range value.Lanes {
		chunks = append(chunks, lane.Baseline.ChunkIDs...)
		chunks = append(chunks, lane.Candidate.ChunkIDs...)
	}
	slices.SortFunc(chunks, artifact.CompareID)
	return slices.Compact(chunks)
}

// Lineage binds the comparison directly to every summary, raw chunk, and
// resource-scope authority used by its materialized aggregates.
func (value ResourceFitnessComparison) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0)
	for _, lane := range value.Lanes {
		for _, endpoint := range []ObservationStream{lane.Baseline, lane.Candidate} {
			parents = append(parents, endpoint.SummaryIDs...)
			parents = append(parents, endpoint.ChunkIDs...)
			parents = append(parents, endpoint.Aggregate.Authorities()...)
		}
	}
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// VerifiedBatch replays every cited source before preparing publication. A
// structurally canonical but forged aggregate can therefore be stored only by
// bypassing this owner, and RequireResourceFitnessComparison will still refuse
// it on read.
func (value ResourceFitnessComparison) VerifiedBatch(
	ctx context.Context,
	reader artifact.Reader,
	key string,
) (artifact.Batch, error) {
	if err := resourceFitnessComparisonCodec.ValidateIdentity(value); err != nil {
		return artifact.Batch{}, err
	}
	if err := verifyResourceFitnessComparisonSources(ctx, reader, value); err != nil {
		return artifact.Batch{}, err
	}
	return resourceFitnessComparisonCodec.Batch(key, value, value.Lineage(), nil)
}

func canonicalizeResourceFitnessComparison(value *ResourceFitnessComparison) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || len(value.Lanes) == 0 ||
		len(value.Lanes) > MaximumAttemptPopulation {
		return errors.New("run record: invalid resource fitness comparison")
	}
	value.Lanes = cloneResourceFitnessLanes(value.Lanes)
	sort.Slice(value.Lanes, func(i, j int) bool { return value.Lanes[i].Name < value.Lanes[j].Name })
	seenAttempts := make(map[artifact.ID]struct{}, len(value.Lanes)+len(value.Lanes))
	strict := false
	for index := range value.Lanes {
		lane := &value.Lanes[index]
		if index > 0 && value.Lanes[index-1].Name == lane.Name {
			return errors.New("run record: duplicate resource fitness lane")
		}
		if err := canonicalizeResourceFitnessLane(lane); err != nil {
			return err
		}
		for _, attempt := range []artifact.ID{lane.Baseline.Scope.Attempt, lane.Candidate.Scope.Attempt} {
			if _, duplicate := seenAttempts[attempt]; duplicate {
				return errors.New("run record: resource fitness attempt is reused across lanes")
			}
			seenAttempts[attempt] = struct{}{}
		}
		strict = strict || len(lane.StrictMetrics) != 0 || lane.StrictInteractions != nil
	}
	value.StrictImprovement = strict
	return nil
}

func canonicalizeResourceFitnessLane(lane *ResourceFitnessLane) error {
	if lane == nil || !validLabel(lane.Name) {
		return errors.New("run record: invalid resource fitness lane")
	}
	lane.RequiredMetrics = slices.Clone(lane.RequiredMetrics)
	slices.Sort(lane.RequiredMetrics)
	for index, metric := range lane.RequiredMetrics {
		if !metric.Valid() || index > 0 && lane.RequiredMetrics[index-1] == metric {
			return fmt.Errorf("run record: invalid required resource metric in lane %q", lane.Name)
		}
	}
	if len(lane.RequiredMetrics) == 0 && !lane.RequireInteractions {
		return fmt.Errorf("run record: resource fitness lane %q has no comparison contract", lane.Name)
	}
	if err := validateMaterializedObservationStream(lane.Baseline); err != nil {
		return errors.Join(err, fmt.Errorf("run record: invalid resource fitness lane %q baseline", lane.Name))
	}
	if err := validateMaterializedObservationStream(lane.Candidate); err != nil {
		return errors.Join(err, fmt.Errorf("run record: invalid resource fitness lane %q candidate", lane.Name))
	}
	baselineScope, candidateScope := lane.Baseline.Scope, lane.Candidate.Scope
	if baselineScope.Surface != candidateScope.Surface || baselineScope.Hardware != candidateScope.Hardware ||
		baselineScope.Provider != candidateScope.Provider || baselineScope.Attempt == candidateScope.Attempt {
		return fmt.Errorf("run record: resource fitness lane %q scopes are not comparable", lane.Name)
	}
	if !sameObservationProtocol(lane.Baseline.Coverage, lane.Candidate.Coverage) {
		return fmt.Errorf("run record: resource fitness lane %q observation protocols differ", lane.Name)
	}
	baselineMeasures, candidateMeasures := lane.Baseline.Aggregate.Measures, lane.Candidate.Aggregate.Measures
	if len(baselineMeasures) != len(candidateMeasures) {
		return fmt.Errorf("run record: resource fitness lane %q metric keysets differ", lane.Name)
	}
	for index := range baselineMeasures {
		if baselineMeasures[index].Metric != candidateMeasures[index].Metric {
			return fmt.Errorf("run record: resource fitness lane %q metric keysets differ", lane.Name)
		}
	}
	for _, required := range lane.RequiredMetrics {
		if _, found := lane.Baseline.Aggregate.Measure(required); !found {
			return fmt.Errorf("run record: resource fitness lane %q lacks required metric %q", lane.Name, required)
		}
	}
	strictMetrics := make([]ResourceMeasure, 0, len(baselineMeasures))
	for index, baseline := range baselineMeasures {
		candidate := candidateMeasures[index]
		if candidate.Value > baseline.Value {
			return fmt.Errorf("run record: resource fitness lane %q regresses metric %q", lane.Name, baseline.Metric)
		}
		if candidate.Value < baseline.Value {
			strictMetrics = append(strictMetrics, ResourceMeasure{
				Metric: baseline.Metric, Value: baseline.Value - candidate.Value,
			})
		}
	}
	lane.StrictMetrics = strictMetrics
	baselineInteractions, candidateInteractions := lane.Baseline.Aggregate.Interactions, lane.Candidate.Aggregate.Interactions
	if (baselineInteractions != nil) != (candidateInteractions != nil) ||
		lane.RequireInteractions && baselineInteractions == nil {
		return fmt.Errorf("run record: resource fitness lane %q interaction coverage differs", lane.Name)
	}
	lane.StrictInteractions = nil
	if baselineInteractions != nil {
		advantage, improved, err := interactionWorkAdvantage(*baselineInteractions, *candidateInteractions)
		if err != nil {
			return errors.Join(err, fmt.Errorf("run record: resource fitness lane %q interaction regression", lane.Name))
		}
		if improved {
			lane.StrictInteractions = &advantage
		}
	}
	return nil
}

// sameObservationProtocol compares only observation denominators. Raw byte
// sizes, chunk boundaries, elapsed timestamps, and aggregate outcome values
// may legitimately differ between otherwise comparable executions.
func sameObservationProtocol(baseline, candidate ObservationStreamCoverage) bool {
	return baseline.Samples == candidate.Samples &&
		baseline.MeasuredSamples == candidate.MeasuredSamples &&
		baseline.HardwareMeasuredSamples == candidate.HardwareMeasuredSamples &&
		baseline.InteractionSamples == candidate.InteractionSamples &&
		slices.Equal(baseline.Metrics, candidate.Metrics) &&
		slices.Equal(baseline.HardwareMetrics, candidate.HardwareMetrics) &&
		slices.Equal(baseline.Kinds, candidate.Kinds)
}

func validateMaterializedObservationStream(stream ObservationStream) error {
	if len(stream.SummaryIDs) == 0 || len(stream.SummaryIDs) != len(stream.ChunkIDs) ||
		len(stream.SummaryIDs) > MaximumAttemptPopulation || stream.Coverage.RawBytes == 0 ||
		stream.Coverage.Samples == 0 || stream.Coverage.FirstOrdinal != traceSequenceStart ||
		stream.Coverage.LastOrdinal != stream.Coverage.Samples ||
		stream.Coverage.LastElapsedNS < stream.Coverage.FirstElapsedNS ||
		stream.Coverage.MeasuredSamples > stream.Coverage.Samples ||
		stream.Coverage.HardwareMeasuredSamples > stream.Coverage.MeasuredSamples ||
		stream.Coverage.InteractionSamples > stream.Coverage.Samples {
		return errors.New("run record: invalid materialized observation stream")
	}
	if err := stream.Aggregate.Validate(); err != nil || stream.Aggregate.Scope != stream.Scope {
		return errors.Join(err, errors.New("run record: materialized observation aggregate differs from scope"))
	}
	seenSummaries := make(map[artifact.ID]struct{}, len(stream.SummaryIDs))
	seenChunks := make(map[artifact.ID]struct{}, len(stream.ChunkIDs))
	for index := range stream.SummaryIDs {
		summary, chunk := stream.SummaryIDs[index], stream.ChunkIDs[index]
		if summary.Kind() != artifact.KindEvidence || chunk.Kind() != artifact.KindFile {
			return errors.New("run record: invalid materialized observation source")
		}
		if _, duplicate := seenSummaries[summary]; duplicate {
			return errors.New("run record: duplicate materialized observation summary")
		}
		if _, duplicate := seenChunks[chunk]; duplicate {
			return errors.New("run record: duplicate materialized observation chunk")
		}
		seenSummaries[summary], seenChunks[chunk] = struct{}{}, struct{}{}
	}
	if err := validateObservationCoverage(stream.Coverage, stream.Aggregate); err != nil {
		return err
	}
	return nil
}

func validateObservationCoverage(coverage ObservationStreamCoverage, aggregate ResourceFitness) error {
	if len(coverage.Metrics) != len(aggregate.Measures) ||
		(coverage.InteractionSamples != 0) != (aggregate.Interactions != nil) {
		return errors.New("run record: materialized observation coverage differs from aggregate")
	}
	for index, count := range coverage.Metrics {
		if !count.Metric.Valid() || count.Samples == 0 || count.Samples > coverage.Samples ||
			index > 0 && coverage.Metrics[index-1].Metric >= count.Metric ||
			aggregate.Measures[index].Metric != count.Metric {
			return errors.New("run record: invalid materialized observation metric coverage")
		}
	}
	for index, count := range coverage.HardwareMetrics {
		globalIndex, found := slices.BinarySearchFunc(coverage.Metrics, count.Metric, func(value ObservationMetricSamples, target ResourceMetric) int {
			return cmp.Compare(value.Metric, target)
		})
		if !count.Metric.Hardware() || count.Samples == 0 || count.Samples > coverage.HardwareMeasuredSamples ||
			index > 0 && coverage.HardwareMetrics[index-1].Metric >= count.Metric || !found ||
			count.Samples > coverage.Metrics[globalIndex].Samples {
			return errors.New("run record: invalid materialized observation hardware coverage")
		}
	}
	var kindTotal uint64
	for index, count := range coverage.Kinds {
		if !slices.Contains(observationSampleKinds, count.Kind) || count.Samples == 0 ||
			index > 0 && coverage.Kinds[index-1].Kind >= count.Kind {
			return errors.New("run record: invalid materialized observation kind coverage")
		}
		var ok bool
		kindTotal, ok = checked.Add64(kindTotal, uint64(count.Samples))
		if !ok {
			return errors.New("run record: materialized observation kind coverage overflows")
		}
	}
	if len(coverage.Kinds) == 0 || kindTotal != uint64(coverage.Samples) {
		return errors.New("run record: materialized observation sample coverage differs")
	}
	return nil
}

func interactionWorkAdvantage(baseline, candidate InteractionWork) (InteractionWork, bool, error) {
	if candidate.SemanticTransitions > baseline.SemanticTransitions ||
		candidate.Commits > baseline.Commits ||
		candidate.ArtifactReads > baseline.ArtifactReads ||
		candidate.BlobReads > baseline.BlobReads ||
		candidate.ScannedFacts > baseline.ScannedFacts ||
		candidate.ReturnedFacts > baseline.ReturnedFacts ||
		candidate.Bytes > baseline.Bytes ||
		candidate.Wakeups > baseline.Wakeups ||
		candidate.Failures > baseline.Failures ||
		candidate.Retries > baseline.Retries ||
		candidate.ModelTurns > baseline.ModelTurns ||
		candidate.ContextBytes > baseline.ContextBytes ||
		candidate.RepeatedContextIDs > baseline.RepeatedContextIDs ||
		candidate.ToolCalls > baseline.ToolCalls ||
		candidate.Waits > baseline.Waits {
		return InteractionWork{}, false, errors.New("run record: candidate interaction work increases")
	}
	advantage := InteractionWork{
		SemanticTransitions: baseline.SemanticTransitions - candidate.SemanticTransitions,
		Commits:             baseline.Commits - candidate.Commits,
		ArtifactReads:       baseline.ArtifactReads - candidate.ArtifactReads,
		BlobReads:           baseline.BlobReads - candidate.BlobReads,
		ScannedFacts:        baseline.ScannedFacts - candidate.ScannedFacts,
		ReturnedFacts:       baseline.ReturnedFacts - candidate.ReturnedFacts,
		Bytes:               baseline.Bytes - candidate.Bytes,
		Wakeups:             baseline.Wakeups - candidate.Wakeups,
		Failures:            baseline.Failures - candidate.Failures,
		Retries:             baseline.Retries - candidate.Retries,
		ModelTurns:          baseline.ModelTurns - candidate.ModelTurns,
		ContextBytes:        baseline.ContextBytes - candidate.ContextBytes,
		RepeatedContextIDs:  baseline.RepeatedContextIDs - candidate.RepeatedContextIDs,
		ToolCalls:           baseline.ToolCalls - candidate.ToolCalls,
		Waits:               baseline.Waits - candidate.Waits,
	}
	return advantage, advantage != (InteractionWork{}), nil
}

func cloneResourceFitnessComparison(value ResourceFitnessComparison) ResourceFitnessComparison {
	value.Lanes = cloneResourceFitnessLanes(value.Lanes)
	return value
}

func cloneResourceFitnessLanes(lanes []ResourceFitnessLane) []ResourceFitnessLane {
	result := slices.Clone(lanes)
	for index := range result {
		result[index].RequiredMetrics = slices.Clone(result[index].RequiredMetrics)
		result[index].Baseline = cloneObservationStream(result[index].Baseline)
		result[index].Candidate = cloneObservationStream(result[index].Candidate)
		result[index].StrictMetrics = slices.Clone(result[index].StrictMetrics)
		if result[index].StrictInteractions != nil {
			work := *result[index].StrictInteractions
			result[index].StrictInteractions = &work
		}
	}
	return result
}

func cloneObservationStream(stream ObservationStream) ObservationStream {
	stream.SummaryIDs = slices.Clone(stream.SummaryIDs)
	stream.ChunkIDs = slices.Clone(stream.ChunkIDs)
	stream.Coverage.Metrics = slices.Clone(stream.Coverage.Metrics)
	stream.Coverage.HardwareMetrics = slices.Clone(stream.Coverage.HardwareMetrics)
	stream.Coverage.Kinds = slices.Clone(stream.Coverage.Kinds)
	stream.Aggregate = cloneResourceFitness(stream.Aggregate)
	return stream
}
