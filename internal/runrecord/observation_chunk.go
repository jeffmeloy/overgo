package runrecord

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/strictjson"
)

const (
	// ObservationChunkMediaType identifies canonical high-volume observation
	// bytes. The raw blob intentionally has no schema: journal projections use
	// the bounded summary document instead of decoding every sample.
	ObservationChunkMediaType = "application/vnd.overgo.observation-chunk+json"
	// ObservationChunkSchema is intentionally empty for raw chunk blobs.
	ObservationChunkSchema = ""
	// ObservationChunkSummaryMediaType identifies the small typed journal fact.
	ObservationChunkSummaryMediaType = "application/vnd.overgo.observation-chunk-summary+json"
	// ObservationChunkSummarySchema identifies the summary document contract.
	ObservationChunkSummarySchema = "overgo/observation-chunk-summary/v1"
	// ObservationChunkAliasRoot scopes one current summary per attempt stream.
	ObservationChunkAliasRoot = "observation/chunk/head/"
	// ObservationChunkMaximumBytes follows the repository's immutable content
	// bound so normalization rejects an oversized request before publication.
	ObservationChunkMaximumBytes = artifact.MaxContentBytes

	observationChunkVersion uint16 = artifact.InitialDocumentVersion
	// This is the shortest canonical valid sample: the interactions pointer
	// preserves an observed zero while the measures list is absent. Dividing the
	// content bound by this exact lower bound caps allocations before encoding;
	// the encoded chunk is checked again because its envelope also consumes bytes.
	observationMinimumSampleEncoding = `{"ordinal":1,"elapsed_ns":0,"kind":"token","interactions":{}}`
	// ObservationChunkMaximumSamples is derived from the repository content
	// limit and the minimum valid canonical sample encoding.
	ObservationChunkMaximumSamples = artifact.MaxContentBytes / len(observationMinimumSampleEncoding)
)

// ObservationSampleKind is a closed class used for bounded per-kind coverage.
type ObservationSampleKind string

const (
	// ObservationSampleToken marks token-level accounting.
	ObservationSampleToken ObservationSampleKind = "token"
	// ObservationSampleMessage marks message-level accounting.
	ObservationSampleMessage ObservationSampleKind = "message"
	// ObservationSampleHardware marks hardware telemetry.
	ObservationSampleHardware ObservationSampleKind = "hardware"
	// ObservationSampleExecution marks execution lifecycle accounting.
	ObservationSampleExecution ObservationSampleKind = "execution"
)

var observationSampleKinds = []ObservationSampleKind{
	ObservationSampleExecution,
	ObservationSampleHardware,
	ObservationSampleMessage,
	ObservationSampleToken,
}

// ObservationSample records one ordered set of resource and interaction
// deltas. ResourceMeasure presence preserves observed zero; absent metrics are
// unknown. Peak host and device measures are observations rather than deltas
// and therefore aggregate by maximum.
type ObservationSample struct {
	Ordinal      uint32                `json:"ordinal"`
	ElapsedNS    uint64                `json:"elapsed_ns"`
	Kind         ObservationSampleKind `json:"kind"`
	Measures     []ResourceMeasure     `json:"measures,omitempty"`
	Interactions *InteractionWork      `json:"interactions,omitempty"`
}

// ObservationChunk is a canonical immutable raw blob. Scope is recorded once
// for the whole ordered sample range, rather than repeated per sample.
type ObservationChunk struct {
	Version  uint16              `json:"version"`
	Scope    ResourceScope       `json:"scope"`
	Previous artifact.ID         `json:"previous,omitzero"`
	Samples  []ObservationSample `json:"samples"`
	ID       artifact.ID         `json:"-"`
}

// ObservationMetricSamples reports how many samples observed one metric.
type ObservationMetricSamples struct {
	Metric  ResourceMetric `json:"metric"`
	Samples uint32         `json:"samples"`
}

// ObservationKindSamples reports how many samples belong to one closed kind.
type ObservationKindSamples struct {
	Kind    ObservationSampleKind `json:"kind"`
	Samples uint32                `json:"samples"`
}

// ObservationChunkStats is the bounded projection metadata for one raw blob.
type ObservationChunkStats struct {
	Bytes              uint64                     `json:"bytes"`
	Samples            uint32                     `json:"samples"`
	FirstOrdinal       uint32                     `json:"first_ordinal"`
	LastOrdinal        uint32                     `json:"last_ordinal"`
	FirstElapsedNS     uint64                     `json:"first_elapsed_ns"`
	LastElapsedNS      uint64                     `json:"last_elapsed_ns"`
	Metrics            []ObservationMetricSamples `json:"metrics,omitempty"`
	Kinds              []ObservationKindSamples   `json:"kinds"`
	InteractionSamples uint32                     `json:"interaction_samples,omitzero"`
}

// ObservationChunkSummary is the small typed fact stored in the journal. Chunk
// is the kind-qualified SHA-256 digest of the exact raw bytes; Aggregate retains
// the complete scope and observed/unknown resource distinction.
type ObservationChunkSummary struct {
	Version   uint16                `json:"version"`
	Chunk     artifact.ID           `json:"chunk"`
	Previous  artifact.ID           `json:"previous,omitzero"`
	Stats     ObservationChunkStats `json:"stats"`
	Aggregate ResourceFitness       `json:"aggregate"`
	ID        artifact.ID           `json:"-"`
}

var observationChunkSummaryCodec = artifact.JSONDocumentCodec(
	"observation chunk summary", artifact.KindEvidence,
	ObservationChunkSummaryMediaType, ObservationChunkSummarySchema,
	canonicalizeObservationChunkSummary,
	func(value ObservationChunkSummary) artifact.ID { return value.ID },
	func(value *ObservationChunkSummary, id artifact.ID) { value.ID = id },
	cloneObservationChunkSummary,
)

// NewObservationChunk canonicalizes and identifies one bounded raw chunk.
func NewObservationChunk(scope ResourceScope, previous artifact.ID, samples []ObservationSample) (ObservationChunk, error) {
	return identifyObservationChunk(ObservationChunk{
		Version: observationChunkVersion, Scope: scope, Previous: previous, Samples: samples,
	})
}

// NewInitialObservationChunk adapts one exact resource observation into the
// first single-sample chunk for its attempt. Producers use this boundary to
// avoid recopying scope, measure, interaction, and initial-ordinal policy.
func NewInitialObservationChunk(
	fitness ResourceFitness,
	kind ObservationSampleKind,
	elapsedNS uint64,
) (ObservationChunk, error) {
	normalized, err := NewResourceFitness(fitness)
	if err != nil {
		return ObservationChunk{}, err
	}
	return NewObservationChunk(normalized.Scope, artifact.ID{}, []ObservationSample{{
		Ordinal: traceSequenceStart, ElapsedNS: elapsedNS, Kind: kind,
		Measures: normalized.Measures, Interactions: normalized.Interactions,
	}})
}

// ParseObservationChunk accepts only the exact canonical raw encoding.
func ParseObservationChunk(data []byte) (ObservationChunk, error) {
	if len(data) == 0 || len(data) > ObservationChunkMaximumBytes {
		return ObservationChunk{}, errors.New("run record: invalid observation chunk byte size")
	}
	var decoded ObservationChunk
	if err := strictjson.DecodeBytes(data, &decoded); err != nil {
		return ObservationChunk{}, fmt.Errorf("run record: decode observation chunk: %w", err)
	}
	identified, canonical, err := canonicalObservationChunk(decoded)
	if err != nil {
		return ObservationChunk{}, err
	}
	if !bytes.Equal(data, canonical) {
		return ObservationChunk{}, errors.New("run record: non-canonical observation chunk")
	}
	if _, err := summarizeObservationChunk(identified, uint64(len(canonical))); err != nil {
		return ObservationChunk{}, err
	}
	return identified, nil
}

// Content returns the exact raw KindFile blob with an intentionally empty
// schema. The summary is the only typed observation document in the journal.
func (value ObservationChunk) Content() (artifact.Content, error) {
	canonical, data, err := canonicalObservationChunk(value)
	if err != nil {
		return artifact.Content{}, err
	}
	if value.ID != canonical.ID || !reflect.DeepEqual(value, canonical) {
		return artifact.Content{}, errors.New("run record: observation chunk is not canonical")
	}
	descriptor := artifact.Descriptor{
		ID: value.ID, Size: uint64(len(data)), MediaType: ObservationChunkMediaType,
		Schema: ObservationChunkSchema,
	}
	content := artifact.Content{Descriptor: descriptor, Data: slices.Clone(data)}
	if err := content.Validate(); err != nil {
		return artifact.Content{}, err
	}
	return content, nil
}

// ValidateIdentity verifies the raw chunk's canonical content identity.
func (value ObservationChunk) ValidateIdentity() error {
	_, err := value.Content()
	return err
}

// SummarizeObservationChunk derives the bounded typed projection without
// publishing it. All arithmetic is checked; overflow is never saturated.
func SummarizeObservationChunk(value ObservationChunk) (ObservationChunkSummary, error) {
	content, err := value.Content()
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	return summarizeObservationChunk(value, uint64(len(content.Data)))
}

// Content returns the canonical summary document.
func (value ObservationChunkSummary) Content() (artifact.Content, error) {
	return observationChunkSummaryCodec.Content(value)
}

// ValidateIdentity verifies the summary document's canonical identity.
func (value ObservationChunkSummary) ValidateIdentity() error {
	return observationChunkSummaryCodec.ValidateIdentity(value)
}

// Lineage binds the summary to its exact raw digest, prior summary, and source
// identities. Raw samples do not add journal edges per event.
func (value ObservationChunkSummary) Lineage() []artifact.Lineage {
	lineage := []artifact.Lineage{{
		Child: value.ID, Parent: value.Chunk, Relation: artifact.RelationContains,
	}}
	if value.Previous.Valid() {
		lineage = append(lineage, artifact.Lineage{
			Child: value.ID, Parent: value.Previous, Relation: artifact.RelationDerivedFrom,
		})
	}
	return append(lineage, artifact.DependencyLineage(value.ID, value.Aggregate.Authorities()...)...)
}

// RequireObservationChunk loads and validates one raw chunk contract.
func RequireObservationChunk(ctx context.Context, reader artifact.Reader, id artifact.ID) (ObservationChunk, error) {
	if ctx == nil || reader == nil || id.Kind() != artifact.KindFile {
		return ObservationChunk{}, errors.New("run record: invalid observation chunk read")
	}
	return requireObservationChunkContent(ctx, reader, id, nil)
}

// requireObservationChunkContent loads one raw chunk and, when supplied,
// refuses a descriptor that changed after a caller's size preflight.
func requireObservationChunkContent(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	preflight *artifact.Descriptor,
) (ObservationChunk, error) {
	descriptor, stream, found, err := reader.OpenContent(ctx, id)
	if err != nil {
		return ObservationChunk{}, err
	}
	if !found {
		return ObservationChunk{}, errors.New("run record: observation chunk is absent")
	}
	if preflight != nil && descriptor != *preflight {
		return ObservationChunk{}, errors.New("run record: observation chunk descriptor changed after preflight")
	}
	content, err := artifact.ReadContentFrom(descriptor, stream)
	if err != nil {
		return ObservationChunk{}, err
	}
	if content.Descriptor.MediaType != ObservationChunkMediaType || content.Descriptor.Schema != ObservationChunkSchema {
		return ObservationChunk{}, errors.New("run record: incompatible observation chunk content")
	}
	value, err := ParseObservationChunk(content.Data)
	if err != nil {
		return ObservationChunk{}, err
	}
	if value.ID != id {
		return ObservationChunk{}, errors.New("run record: observation chunk identity mismatch")
	}
	return value, nil
}

// RequireObservationChunkSummary loads a typed summary, then loads and
// recomputes its raw chunk. A typed fact with absent or differing blob bytes is
// not accepted as observation evidence.
func RequireObservationChunkSummary(ctx context.Context, reader artifact.Reader, id artifact.ID) (ObservationChunkSummary, error) {
	value, _, err := requireObservationChunkSummaryAndChunk(ctx, reader, id)
	return value, err
}

func requireObservationChunkSummaryAndChunk(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (ObservationChunkSummary, ObservationChunk, error) {
	value, err := observationChunkSummaryCodec.Require(ctx, reader, id)
	if err != nil {
		return ObservationChunkSummary{}, ObservationChunk{}, err
	}
	chunk, err := RequireObservationChunk(ctx, reader, value.Chunk)
	if err != nil {
		return ObservationChunkSummary{}, ObservationChunk{}, err
	}
	if err := VerifyObservationChunk(value, chunk); err != nil {
		return ObservationChunkSummary{}, ObservationChunk{}, err
	}
	return value, chunk, nil
}

// VerifyObservationChunk recomputes digest, byte count, counters, aggregates,
// and canonical summary identity from the raw blob.
func VerifyObservationChunk(summary ObservationChunkSummary, chunk ObservationChunk) error {
	if err := summary.ValidateIdentity(); err != nil {
		return err
	}
	want, err := SummarizeObservationChunk(chunk)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(summary, want) {
		return errors.New("run record: observation chunk summary differs from raw content")
	}
	return nil
}

// BindObservationChunk appends one raw chunk, verified summary, lineage, and
// attempt-head compare-and-set update to an existing semantic batch.
// Continuations load the exact prior summary named by the chunk; the alias CAS
// then refuses a stale sibling. The repository remains the single owner of
// final request sizing and atomic admission after producers compose the batch.
func BindObservationChunk(
	ctx context.Context,
	reader artifact.Reader,
	batch *artifact.Batch,
	value ObservationChunk,
) (ObservationChunkSummary, error) {
	if ctx == nil || reader == nil || batch == nil {
		return ObservationChunkSummary{}, errors.New("run record: observation chunk dependencies are absent")
	}
	summary, err := SummarizeObservationChunk(value)
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	var head ObservationChunkSummary
	hasHead := value.Previous.Valid()
	if hasHead {
		head, err = RequireObservationChunkSummary(ctx, reader, value.Previous)
		if err != nil {
			return ObservationChunkSummary{}, err
		}
	}
	if err := validateObservationChunkContinuation(value, head, hasHead); err != nil {
		return ObservationChunkSummary{}, err
	}
	rawContent, err := value.Content()
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	summaryContent, err := summary.Content()
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	alias := artifact.AliasBinding{
		Name: ObservationChunkAlias(value.Scope.Attempt), Target: summary.ID,
	}
	if hasHead {
		alias.Previous = artifact.IDPointer(head.ID)
	}
	candidate := *batch
	candidate.Artifacts = slices.Clone(batch.Artifacts)
	candidate.Contents = append(slices.Clone(batch.Contents), rawContent, summaryContent)
	candidate.Lineage = append(slices.Clone(batch.Lineage), summary.Lineage()...)
	candidate.Aliases = append(artifact.CloneAliasBindings(batch.Aliases), alias)
	if err := candidate.Validate(); err != nil {
		return ObservationChunkSummary{}, err
	}
	*batch = candidate
	return summary, nil
}

// ObservationChunkAlias returns the sole mutable head for one attempt stream.
func ObservationChunkAlias(attempt artifact.ID) string {
	return ObservationChunkAliasRoot + attempt.String()
}

func identifyObservationChunk(value ObservationChunk) (ObservationChunk, error) {
	identified, data, err := canonicalObservationChunk(value)
	if err != nil {
		return ObservationChunk{}, err
	}
	if _, err := summarizeObservationChunk(identified, uint64(len(data))); err != nil {
		return ObservationChunk{}, err
	}
	return identified, nil
}

func canonicalObservationChunk(value ObservationChunk) (ObservationChunk, []byte, error) {
	if len(value.Samples) == 0 || len(value.Samples) > ObservationChunkMaximumSamples {
		return ObservationChunk{}, nil, errors.New("run record: invalid observation chunk sample count")
	}
	value = cloneObservationChunk(value)
	value.ID = artifact.ID{}
	if err := canonicalizeObservationChunk(&value); err != nil {
		return ObservationChunk{}, nil, err
	}
	data, err := json.Marshal(value)
	if err != nil {
		return ObservationChunk{}, nil, err
	}
	if len(data) == 0 || len(data) > ObservationChunkMaximumBytes {
		return ObservationChunk{}, nil, errors.New("run record: observation chunk exceeds content byte bound")
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return ObservationChunk{}, nil, err
	}
	value.ID = id
	return value, data, nil
}

func canonicalizeObservationChunk(value *ObservationChunk) error {
	if value == nil || value.Version != observationChunkVersion || len(value.Samples) == 0 ||
		len(value.Samples) > ObservationChunkMaximumSamples ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid observation chunk envelope")
	}
	if err := validateObservationScope(value.Scope); err != nil {
		return err
	}
	first := value.Samples[0].Ordinal
	if first == traceSequenceStart {
		if value.Previous.Valid() {
			return errors.New("run record: initial observation chunk has previous summary")
		}
	} else if first < traceSequenceStart || !value.Previous.Valid() {
		return errors.New("run record: observation continuation is missing previous summary")
	}
	for index := range value.Samples {
		sample := &value.Samples[index]
		want := uint64(first) + uint64(index)
		if want > math.MaxUint32 || sample.Ordinal != uint32(want) ||
			index > 0 && sample.ElapsedNS < value.Samples[index-1].ElapsedNS ||
			!slices.Contains(observationSampleKinds, sample.Kind) {
			return errors.New("run record: invalid observation sample order or kind")
		}
		sample.Measures = slices.Clone(sample.Measures)
		for _, measure := range sample.Measures {
			if !validResourceMetric(measure.Metric) {
				return errors.New("run record: foreign observation resource metric")
			}
		}
		sort.Slice(sample.Measures, func(i, j int) bool { return sample.Measures[i].Metric < sample.Measures[j].Metric })
		if duplicateResourceMetric(sample.Measures) {
			return errors.New("run record: duplicate observation resource metric")
		}
		if len(sample.Measures) == 0 {
			sample.Measures = nil
		}
		if sample.Interactions != nil {
			work := *sample.Interactions
			sample.Interactions = &work
		}
		if len(sample.Measures) == 0 && sample.Interactions == nil {
			return errors.New("run record: observation sample records no observations")
		}
	}
	return nil
}

func validateObservationScope(scope ResourceScope) error {
	if err := validateResourceScope(scope); err != nil {
		return errors.Join(errors.New("run record: invalid observation scope"), err)
	}
	return nil
}

func summarizeObservationChunk(value ObservationChunk, byteCount uint64) (ObservationChunkSummary, error) {
	metricCounts := make(map[ResourceMetric]uint32)
	metricTotals := make(map[ResourceMetric]uint64)
	kindCounts := make(map[ObservationSampleKind]uint32)
	var interactions InteractionWork
	var interactionSamples uint32
	for _, sample := range value.Samples {
		kindCounts[sample.Kind]++
		for _, measure := range sample.Measures {
			metricCounts[measure.Metric]++
		}
		if !addResourceMeasures(metricTotals, sample.Measures) {
			return ObservationChunkSummary{}, errors.New("run record: observation resource aggregate overflows")
		}
		if sample.Interactions != nil {
			interactionSamples++
			if !addInteractionWork(&interactions, *sample.Interactions) {
				return ObservationChunkSummary{}, errors.New("run record: observation interaction aggregate overflows")
			}
		}
	}
	measures := make([]ResourceMeasure, 0, len(metricTotals))
	metricSamples := make([]ObservationMetricSamples, 0, len(metricCounts))
	for metric, value := range metricTotals {
		measures = append(measures, ResourceMeasure{Metric: metric, Value: value})
		metricSamples = append(metricSamples, ObservationMetricSamples{Metric: metric, Samples: metricCounts[metric]})
	}
	kindSamples := make([]ObservationKindSamples, 0, len(kindCounts))
	for kind, count := range kindCounts {
		kindSamples = append(kindSamples, ObservationKindSamples{Kind: kind, Samples: count})
	}
	var observedInteractions *InteractionWork
	if interactionSamples != 0 {
		observedInteractions = &interactions
	}
	aggregate, err := NewResourceFitness(ResourceFitness{
		Scope: value.Scope, Measures: measures, Interactions: observedInteractions,
	})
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	first, last := value.Samples[0], value.Samples[len(value.Samples)-1]
	summary := ObservationChunkSummary{
		Version: observationChunkVersion, Chunk: value.ID, Previous: value.Previous,
		Stats: ObservationChunkStats{
			Bytes: byteCount, Samples: uint32(len(value.Samples)),
			FirstOrdinal: first.Ordinal, LastOrdinal: last.Ordinal,
			FirstElapsedNS: first.ElapsedNS, LastElapsedNS: last.ElapsedNS,
			Metrics: metricSamples, Kinds: kindSamples, InteractionSamples: interactionSamples,
		},
		Aggregate: aggregate,
	}
	return observationChunkSummaryCodec.New(summary)
}

func canonicalizeObservationChunkSummary(value *ObservationChunkSummary) error {
	if value == nil || value.Version != observationChunkVersion || value.Chunk.Kind() != artifact.KindFile ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence ||
		value.Stats.Bytes == 0 || value.Stats.Bytes > ObservationChunkMaximumBytes || value.Stats.Samples == 0 ||
		uint64(value.Stats.Samples) > uint64(ObservationChunkMaximumSamples) ||
		value.Stats.FirstOrdinal < traceSequenceStart || value.Stats.LastOrdinal < value.Stats.FirstOrdinal ||
		uint64(value.Stats.LastOrdinal)-uint64(value.Stats.FirstOrdinal)+1 != uint64(value.Stats.Samples) ||
		value.Stats.LastElapsedNS < value.Stats.FirstElapsedNS ||
		(value.Stats.FirstOrdinal == traceSequenceStart) == value.Previous.Valid() {
		return errors.New("run record: invalid observation chunk summary")
	}
	aggregate, err := NewResourceFitness(value.Aggregate)
	if err != nil {
		return errors.Join(errors.New("run record: invalid observation chunk aggregate"), err)
	}
	value.Aggregate = aggregate
	value.Stats.Metrics = slices.Clone(value.Stats.Metrics)
	sort.Slice(value.Stats.Metrics, func(i, j int) bool { return value.Stats.Metrics[i].Metric < value.Stats.Metrics[j].Metric })
	for index, count := range value.Stats.Metrics {
		_, observed := value.Aggregate.Measure(count.Metric)
		if !validResourceMetric(count.Metric) || count.Samples == 0 || count.Samples > value.Stats.Samples || !observed ||
			index > 0 && value.Stats.Metrics[index-1].Metric == count.Metric {
			return errors.New("run record: invalid observation metric sample counts")
		}
	}
	if len(value.Stats.Metrics) != len(value.Aggregate.Measures) {
		return errors.New("run record: observation metric coverage differs from aggregate")
	}
	if len(value.Stats.Metrics) == 0 {
		value.Stats.Metrics = nil
	}
	value.Stats.Kinds = slices.Clone(value.Stats.Kinds)
	sort.Slice(value.Stats.Kinds, func(i, j int) bool { return value.Stats.Kinds[i].Kind < value.Stats.Kinds[j].Kind })
	var kindTotal uint64
	for index, count := range value.Stats.Kinds {
		if !slices.Contains(observationSampleKinds, count.Kind) || count.Samples == 0 ||
			index > 0 && value.Stats.Kinds[index-1].Kind == count.Kind {
			return errors.New("run record: invalid observation kind sample counts")
		}
		kindTotal += uint64(count.Samples)
	}
	if len(value.Stats.Kinds) == 0 || kindTotal != uint64(value.Stats.Samples) ||
		value.Stats.InteractionSamples > value.Stats.Samples ||
		(value.Stats.InteractionSamples != 0) != (value.Aggregate.Interactions != nil) {
		return errors.New("run record: observation sample coverage differs")
	}
	return nil
}

func validateObservationChunkContinuation(value ObservationChunk, head ObservationChunkSummary, found bool) error {
	first := value.Samples[0]
	if !found {
		if value.Previous.Valid() || first.Ordinal != traceSequenceStart {
			return errors.New("run record: observation stream must start at initial ordinal")
		}
		return nil
	}
	if value.Previous != head.ID || value.Scope != head.Aggregate.Scope || head.Stats.LastOrdinal == math.MaxUint32 ||
		first.Ordinal != head.Stats.LastOrdinal+1 || first.ElapsedNS < head.Stats.LastElapsedNS {
		return errors.New("run record: observation chunk continuation differs")
	}
	return nil
}

func cloneObservationChunk(value ObservationChunk) ObservationChunk {
	value.Samples = slices.Clone(value.Samples)
	for index := range value.Samples {
		value.Samples[index].Measures = slices.Clone(value.Samples[index].Measures)
		if value.Samples[index].Interactions != nil {
			work := *value.Samples[index].Interactions
			value.Samples[index].Interactions = &work
		}
	}
	return value
}

func cloneObservationChunkSummary(value ObservationChunkSummary) ObservationChunkSummary {
	value.Stats.Metrics = slices.Clone(value.Stats.Metrics)
	value.Stats.Kinds = slices.Clone(value.Stats.Kinds)
	value.Aggregate = cloneResourceFitness(value.Aggregate)
	return value
}

func addInteractionWork(total *InteractionWork, delta InteractionWork) bool {
	values := []struct {
		total *uint64
		delta uint64
	}{
		{&total.SemanticTransitions, delta.SemanticTransitions},
		{&total.Commits, delta.Commits},
		{&total.ArtifactReads, delta.ArtifactReads},
		{&total.BlobReads, delta.BlobReads},
		{&total.ScannedFacts, delta.ScannedFacts},
		{&total.ReturnedFacts, delta.ReturnedFacts},
		{&total.Bytes, delta.Bytes},
		{&total.Wakeups, delta.Wakeups},
		{&total.Failures, delta.Failures},
		{&total.Retries, delta.Retries},
		{&total.ModelTurns, delta.ModelTurns},
		{&total.ContextBytes, delta.ContextBytes},
		{&total.RepeatedContextIDs, delta.RepeatedContextIDs},
		{&total.ToolCalls, delta.ToolCalls},
		{&total.Waits, delta.Waits},
	}
	for _, value := range values {
		total, ok := checked.Add64(*value.total, value.delta)
		if !ok {
			return false
		}
		*value.total = total
	}
	return true
}

func addResourceMeasures(total map[ResourceMetric]uint64, delta []ResourceMeasure) bool {
	for _, measure := range delta {
		current, observed := total[measure.Metric]
		if measure.Metric == ResourcePeakHostBytes || measure.Metric == ResourcePeakDeviceBytes {
			if !observed || measure.Value > current {
				total[measure.Metric] = measure.Value
			}
			continue
		}
		next, ok := checked.Add64(current, measure.Value)
		if !ok {
			return false
		}
		total[measure.Metric] = next
	}
	return true
}
