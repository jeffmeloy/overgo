package runrecord

import (
	"context"
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

// ObservationStreamCoverage reports exact sample presence across an attempt's
// complete chunk stream. MeasuredSamples counts the union of samples carrying
// at least one resource measure; observed zero measures remain included.
type ObservationStreamCoverage struct {
	RawBytes                uint64                     `json:"raw_bytes"`
	Samples                 uint32                     `json:"samples"`
	MeasuredSamples         uint32                     `json:"measured_samples"`
	HardwareMeasuredSamples uint32                     `json:"hardware_measured_samples"`
	FirstOrdinal            uint32                     `json:"first_ordinal"`
	LastOrdinal             uint32                     `json:"last_ordinal"`
	FirstElapsedNS          uint64                     `json:"first_elapsed_ns"`
	LastElapsedNS           uint64                     `json:"last_elapsed_ns"`
	Metrics                 []ObservationMetricSamples `json:"metrics,omitempty"`
	HardwareMetrics         []ObservationMetricSamples `json:"hardware_metrics,omitempty"`
	Kinds                   []ObservationKindSamples   `json:"kinds"`
	InteractionSamples      uint32                     `json:"interaction_samples,omitempty"`
}

// ObservationStreamBounds limits one complete verified stream load. Raw bytes
// are counted from immutable descriptors before their contents are opened.
type ObservationStreamBounds struct {
	MaxChunks   int    `json:"max_chunks"`
	MaxRawBytes uint64 `json:"max_raw_bytes"`
}

// ObservationStream is a verified oldest-to-newest view of the current alias
// for one attempt. The final SummaryIDs entry is the exact alias target.
type ObservationStream struct {
	Scope      ResourceScope             `json:"scope"`
	SummaryIDs []artifact.ID             `json:"summary_ids"`
	ChunkIDs   []artifact.ID             `json:"chunk_ids"`
	Coverage   ObservationStreamCoverage `json:"coverage"`
	Aggregate  ResourceFitness           `json:"aggregate"`
}

type observationStreamPair struct {
	summary  ObservationChunkSummary
	chunk    ObservationChunk
	rawBytes uint64
}

// LoadObservationStream resolves and verifies one complete attempt stream.
// Bounds are caller-owned interaction limits; an oversized stream is refused
// rather than truncated into an inexact aggregate.
func LoadObservationStream(
	ctx context.Context,
	reader artifact.Reader,
	attempt artifact.ID,
	bounds ObservationStreamBounds,
) (ObservationStream, bool, error) {
	if ctx == nil || reader == nil || bounds.MaxChunks < 0 ||
		attempt.Kind() != artifact.KindRun && attempt.Kind() != artifact.KindEvidence {
		return ObservationStream{}, false, errors.New("run record: invalid observation stream query")
	}
	head, found, err := artifact.ResolveAlias(ctx, reader, ObservationChunkAlias(attempt))
	if err != nil || !found {
		return ObservationStream{}, found, err
	}
	stream, err := RequireObservationStream(ctx, reader, attempt, head, bounds)
	return stream, true, err
}

// RequireObservationStream verifies the complete immutable stream ending at
// summaryHead. Unlike LoadObservationStream it never consults the mutable
// attempt alias, so evidence remains reproducible after newer chunks advance
// that alias.
func RequireObservationStream(
	ctx context.Context,
	reader artifact.Reader,
	attempt artifact.ID,
	summaryHead artifact.ID,
	bounds ObservationStreamBounds,
) (ObservationStream, error) {
	if ctx == nil || reader == nil || bounds.MaxChunks < 0 ||
		(attempt.Kind() != artifact.KindRun && attempt.Kind() != artifact.KindEvidence) ||
		summaryHead.Kind() != artifact.KindEvidence {
		return ObservationStream{}, errors.New("run record: invalid exact observation stream query")
	}
	if bounds.MaxChunks == 0 {
		return ObservationStream{}, errors.New("run record: observation stream exceeds chunk read bound")
	}
	if bounds.MaxRawBytes == 0 {
		return ObservationStream{}, errors.New("run record: observation stream exceeds raw byte read bound")
	}
	var rawBytes uint64
	pair, rawBytes, err := requireObservationStreamPair(ctx, reader, summaryHead, rawBytes, bounds.MaxRawBytes)
	if err != nil {
		return ObservationStream{}, err
	}
	if pair.summary.Aggregate.Scope.Attempt != attempt {
		return ObservationStream{}, errors.New("run record: observation stream head changes attempt")
	}

	reverse := []observationStreamPair{pair}
	seen := map[artifact.ID]struct{}{pair.summary.ID: {}}
	ordinalBound := uint64(pair.summary.Stats.LastOrdinal)
	for pair.chunk.Previous.Valid() {
		if len(reverse) >= bounds.MaxChunks {
			return ObservationStream{}, errors.New("run record: observation stream exceeds chunk read bound")
		}
		if uint64(len(reverse)) >= ordinalBound {
			return ObservationStream{}, errors.New("run record: observation stream exceeds ordinal integrity bound")
		}
		if _, duplicate := seen[pair.chunk.Previous]; duplicate {
			return ObservationStream{}, errors.New("run record: observation stream contains summary cycle")
		}
		prior, nextRawBytes, requireErr := requireObservationStreamPair(
			ctx, reader, pair.chunk.Previous, rawBytes, bounds.MaxRawBytes,
		)
		if requireErr != nil {
			return ObservationStream{}, requireErr
		}
		if err := validateObservationChunkContinuation(pair.chunk, prior.summary, true); err != nil {
			return ObservationStream{}, err
		}
		seen[prior.summary.ID] = struct{}{}
		reverse = append(reverse, prior)
		pair = prior
		rawBytes = nextRawBytes
	}
	if err := validateObservationChunkContinuation(pair.chunk, ObservationChunkSummary{}, false); err != nil {
		return ObservationStream{}, err
	}
	slices.Reverse(reverse)
	stream, err := aggregateObservationStream(reverse)
	if err == nil && stream.Coverage.RawBytes != rawBytes {
		return ObservationStream{}, errors.New("run record: observation stream raw byte count changed after preflight")
	}
	return stream, err
}

func requireObservationStreamPair(
	ctx context.Context,
	reader artifact.Reader,
	summaryID artifact.ID,
	rawBytes uint64,
	maxRawBytes uint64,
) (observationStreamPair, uint64, error) {
	summary, err := observationChunkSummaryCodec.Require(ctx, reader, summaryID)
	if err != nil {
		return observationStreamPair{}, rawBytes, err
	}
	descriptor, found, err := reader.Artifact(ctx, summary.Chunk)
	if err != nil {
		return observationStreamPair{}, rawBytes, err
	}
	if !found {
		return observationStreamPair{}, rawBytes, errors.New("run record: observation chunk descriptor is absent")
	}
	if descriptor.ID != summary.Chunk || descriptor.Size == 0 || descriptor.Size > ObservationChunkMaximumBytes ||
		descriptor.MediaType != ObservationChunkMediaType || descriptor.Schema != ObservationChunkSchema {
		return observationStreamPair{}, rawBytes, errors.New("run record: incompatible observation chunk descriptor")
	}
	nextRawBytes, ok := checked.Add64(rawBytes, descriptor.Size)
	if !ok || nextRawBytes > maxRawBytes {
		return observationStreamPair{}, rawBytes, errors.New("run record: observation stream exceeds raw byte read bound")
	}
	chunk, err := requireObservationChunkContent(ctx, reader, summary.Chunk, &descriptor)
	if err != nil {
		return observationStreamPair{}, rawBytes, err
	}
	if err := VerifyObservationChunk(summary, chunk); err != nil {
		return observationStreamPair{}, rawBytes, err
	}
	return observationStreamPair{summary: summary, chunk: chunk, rawBytes: descriptor.Size}, nextRawBytes, nil
}

func aggregateObservationStream(pairs []observationStreamPair) (ObservationStream, error) {
	first := pairs[0]
	last := pairs[len(pairs)-1]
	stream := ObservationStream{
		Scope:      first.summary.Aggregate.Scope,
		SummaryIDs: make([]artifact.ID, 0, len(pairs)),
		ChunkIDs:   make([]artifact.ID, 0, len(pairs)),
		Coverage: ObservationStreamCoverage{
			FirstOrdinal:   first.summary.Stats.FirstOrdinal,
			LastOrdinal:    last.summary.Stats.LastOrdinal,
			FirstElapsedNS: first.summary.Stats.FirstElapsedNS,
			LastElapsedNS:  last.summary.Stats.LastElapsedNS,
		},
	}
	measureTotals := make(map[ResourceMetric]uint64)
	metricSamples := make(map[ResourceMetric]uint64)
	hardwareMetricSamples := make(map[ResourceMetric]uint64)
	kindSamples := make(map[ObservationSampleKind]uint64)
	var interactionTotal InteractionWork
	var sampleTotal, measuredTotal, hardwareMeasuredTotal, interactionSamples uint64
	for _, pair := range pairs {
		stream.SummaryIDs = append(stream.SummaryIDs, pair.summary.ID)
		stream.ChunkIDs = append(stream.ChunkIDs, pair.summary.Chunk)
		if pair.summary.Aggregate.Scope != stream.Scope {
			return ObservationStream{}, errors.New("run record: observation stream changes resource scope")
		}
		var ok bool
		stream.Coverage.RawBytes, ok = checked.Add64(stream.Coverage.RawBytes, pair.rawBytes)
		if !ok {
			return ObservationStream{}, errors.New("run record: observation stream byte count overflows")
		}
		sampleTotal, ok = checked.Add64(sampleTotal, uint64(pair.summary.Stats.Samples))
		if !ok {
			return ObservationStream{}, errors.New("run record: observation stream sample count overflows")
		}
		for _, sample := range pair.chunk.Samples {
			if len(sample.Measures) != 0 {
				measuredTotal++
			}
			if sample.Kind == ObservationSampleHardware {
				hardwareObserved := false
				for _, measure := range sample.Measures {
					if measure.Metric.Hardware() {
						hardwareMetricSamples[measure.Metric]++
						hardwareObserved = true
					}
				}
				if hardwareObserved {
					hardwareMeasuredTotal++
				}
			}
		}
		for _, count := range pair.summary.Stats.Metrics {
			metricSamples[count.Metric], ok = checked.Add64(metricSamples[count.Metric], uint64(count.Samples))
			if !ok {
				return ObservationStream{}, errors.New("run record: observation metric sample count overflows")
			}
		}
		for _, count := range pair.summary.Stats.Kinds {
			kindSamples[count.Kind], ok = checked.Add64(kindSamples[count.Kind], uint64(count.Samples))
			if !ok {
				return ObservationStream{}, errors.New("run record: observation kind sample count overflows")
			}
		}
		interactionSamples, ok = checked.Add64(interactionSamples, uint64(pair.summary.Stats.InteractionSamples))
		if !ok || !addResourceMeasures(measureTotals, pair.summary.Aggregate.Measures) {
			return ObservationStream{}, errors.New("run record: observation stream aggregate overflows")
		}
		if pair.summary.Aggregate.Interactions != nil &&
			!addInteractionWork(&interactionTotal, *pair.summary.Aggregate.Interactions) {
			return ObservationStream{}, errors.New("run record: observation stream interaction aggregate overflows")
		}
	}
	if sampleTotal > math.MaxUint32 || measuredTotal > sampleTotal || hardwareMeasuredTotal > measuredTotal ||
		interactionSamples > sampleTotal ||
		sampleTotal != uint64(stream.Coverage.LastOrdinal) {
		return ObservationStream{}, errors.New("run record: observation stream coverage differs from ordinal span")
	}
	stream.Coverage.Samples = uint32(sampleTotal)
	stream.Coverage.MeasuredSamples = uint32(measuredTotal)
	stream.Coverage.HardwareMeasuredSamples = uint32(hardwareMeasuredTotal)
	stream.Coverage.InteractionSamples = uint32(interactionSamples)
	for metric, count := range metricSamples {
		if count > sampleTotal {
			return ObservationStream{}, errors.New("run record: observation stream metric coverage exceeds samples")
		}
		stream.Coverage.Metrics = append(stream.Coverage.Metrics, ObservationMetricSamples{
			Metric: metric, Samples: uint32(count),
		})
	}
	for metric, count := range hardwareMetricSamples {
		if count > hardwareMeasuredTotal {
			return ObservationStream{}, errors.New("run record: hardware metric coverage exceeds samples")
		}
		stream.Coverage.HardwareMetrics = append(stream.Coverage.HardwareMetrics, ObservationMetricSamples{
			Metric: metric, Samples: uint32(count),
		})
	}
	for kind, count := range kindSamples {
		if count > sampleTotal {
			return ObservationStream{}, errors.New("run record: observation stream kind coverage exceeds samples")
		}
		stream.Coverage.Kinds = append(stream.Coverage.Kinds, ObservationKindSamples{
			Kind: kind, Samples: uint32(count),
		})
	}
	sort.Slice(stream.Coverage.Metrics, func(i, j int) bool {
		return stream.Coverage.Metrics[i].Metric < stream.Coverage.Metrics[j].Metric
	})
	sort.Slice(stream.Coverage.HardwareMetrics, func(i, j int) bool {
		return stream.Coverage.HardwareMetrics[i].Metric < stream.Coverage.HardwareMetrics[j].Metric
	})
	sort.Slice(stream.Coverage.Kinds, func(i, j int) bool {
		return stream.Coverage.Kinds[i].Kind < stream.Coverage.Kinds[j].Kind
	})
	measures := make([]ResourceMeasure, 0, len(measureTotals))
	for metric, value := range measureTotals {
		measures = append(measures, ResourceMeasure{Metric: metric, Value: value})
	}
	var interactions *InteractionWork
	if interactionSamples != 0 {
		interactions = &interactionTotal
	}
	aggregate, err := NewResourceFitness(ResourceFitness{
		Scope: stream.Scope, Measures: measures, Interactions: interactions,
	})
	if err != nil {
		return ObservationStream{}, err
	}
	stream.Aggregate = aggregate
	return stream, nil
}
