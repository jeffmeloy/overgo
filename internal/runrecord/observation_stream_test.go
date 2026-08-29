package runrecord

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

type observationStreamOverrideReader struct {
	artifact.Reader
	aliases  map[string]artifact.ID
	contents map[artifact.ID]artifact.Content
	opened   map[artifact.ID]int
}

func (r observationStreamOverrideReader) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	if id, found := r.aliases[name]; found {
		return id, true, nil
	}
	return r.Reader.ResolveAlias(ctx, name)
}

func (r observationStreamOverrideReader) OpenContent(
	ctx context.Context,
	id artifact.ID,
) (artifact.Descriptor, io.Reader, bool, error) {
	if r.opened != nil {
		r.opened[id]++
	}
	if content, found := r.contents[id]; found {
		return content.Descriptor, bytes.NewReader(content.Data), true, nil
	}
	return r.Reader.OpenContent(ctx, id)
}

func TestLoadObservationStream(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	id := func(kind artifact.Kind, label string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, label)
	}
	scope := ResourceScope{
		Surface:  SurfaceServing,
		Model:    id(artifact.KindModel, "stream-model"),
		Hardware: id(artifact.KindEvidence, "stream-hardware"),
		Provider: id(artifact.KindProfile, "stream-provider"),
		Workload: id(artifact.KindRecipe, "stream-workload"),
		Attempt:  id(artifact.KindRun, "stream-attempt"),
	}
	authorities := scopeAuthorities(scope)
	descriptors := make([]artifact.Descriptor, len(authorities))
	for index, authority := range authorities {
		descriptors[index] = artifact.Descriptor{ID: authority}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "observation/stream/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}

	observedZero := InteractionWork{}
	first, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{
		{Ordinal: 1, ElapsedNS: 10, Kind: ObservationSampleToken, Measures: []ResourceMeasure{
			{Metric: ResourceCostUnits, Value: 0},
			{Metric: ResourcePeakHostBytes, Value: 100},
			{Metric: ResourceWallNS, Value: 5},
		}},
		{Ordinal: 2, ElapsedNS: 20, Kind: ObservationSampleHardware, Measures: []ResourceMeasure{
			{Metric: ResourcePeakHostBytes, Value: 80},
		}, Interactions: &observedZero},
	})
	if err != nil {
		t.Fatal(err)
	}
	firstSummary, err := commitObservationChunk(t, ctx, store, first)
	if err != nil {
		t.Fatal(err)
	}
	observedWork := InteractionWork{Failures: 1, Retries: 2}
	second, err := NewObservationChunk(scope, firstSummary.ID, []ObservationSample{
		{Ordinal: 3, ElapsedNS: 30, Kind: ObservationSampleMessage, Measures: []ResourceMeasure{
			{Metric: ResourcePeakHostBytes, Value: 120},
			{Metric: ResourceWallNS, Value: 7},
		}, Interactions: &observedWork},
	})
	if err != nil {
		t.Fatal(err)
	}
	secondSummary, err := commitObservationChunk(t, ctx, store, second)
	if err != nil {
		t.Fatal(err)
	}

	wantSummaryIDs := []artifact.ID{firstSummary.ID, secondSummary.ID}
	wantChunkIDs := []artifact.ID{firstSummary.Chunk, secondSummary.Chunk}
	wantBytes := firstSummary.Stats.Bytes + secondSummary.Stats.Bytes
	bounds := ObservationStreamBounds{MaxChunks: len(wantSummaryIDs), MaxRawBytes: wantBytes}
	stream, found, err := LoadObservationStream(ctx, store, scope.Attempt, bounds)
	if err != nil || !found {
		t.Fatalf("load = (%+v, %v, %v)", stream, found, err)
	}
	if stream.Scope != scope || !slices.Equal(stream.SummaryIDs, wantSummaryIDs) ||
		!slices.Equal(stream.ChunkIDs, wantChunkIDs) {
		t.Fatalf("ordered stream = %+v", stream)
	}
	wantMetrics := []ObservationMetricSamples{
		{Metric: ResourceCostUnits, Samples: 1},
		{Metric: ResourcePeakHostBytes, Samples: 3},
		{Metric: ResourceWallNS, Samples: 2},
	}
	wantKinds := []ObservationKindSamples{
		{Kind: ObservationSampleHardware, Samples: 1},
		{Kind: ObservationSampleMessage, Samples: 1},
		{Kind: ObservationSampleToken, Samples: 1},
	}
	wantHardwareMetrics := []ObservationMetricSamples{{Metric: ResourcePeakHostBytes, Samples: 1}}
	if stream.Coverage.RawBytes != wantBytes || stream.Coverage.Samples != 3 ||
		stream.Coverage.MeasuredSamples != 3 || stream.Coverage.HardwareMeasuredSamples != 1 ||
		stream.Coverage.InteractionSamples != 2 ||
		stream.Coverage.FirstOrdinal != 1 || stream.Coverage.LastOrdinal != 3 ||
		stream.Coverage.FirstElapsedNS != 10 || stream.Coverage.LastElapsedNS != 30 ||
		!slices.Equal(stream.Coverage.Metrics, wantMetrics) ||
		!slices.Equal(stream.Coverage.HardwareMetrics, wantHardwareMetrics) ||
		!slices.Equal(stream.Coverage.Kinds, wantKinds) {
		t.Fatalf("stream coverage = %+v", stream.Coverage)
	}
	for metric, want := range map[ResourceMetric]uint64{
		ResourceCostUnits: 0, ResourcePeakHostBytes: 120, ResourceWallNS: 12,
	} {
		if got, observed := stream.Aggregate.Measure(metric); !observed || got != want {
			t.Fatalf("aggregate %s = (%d, %v), want (%d, true)", metric, got, observed, want)
		}
	}
	if stream.Aggregate.Interactions == nil || *stream.Aggregate.Interactions != observedWork {
		t.Fatalf("aggregate interactions = %+v", stream.Aggregate.Interactions)
	}

	if _, found, err := LoadObservationStream(
		ctx, store, id(artifact.KindRun, "missing-stream"), bounds,
	); err != nil || found {
		t.Fatalf("missing alias = (%v, %v)", found, err)
	}
	if _, found, err := LoadObservationStream(ctx, store, scope.Attempt, ObservationStreamBounds{
		MaxChunks: len(wantSummaryIDs) - 1, MaxRawBytes: wantBytes,
	}); err == nil || !found {
		t.Fatalf("chunk bound refusal = (%v, %v)", found, err)
	}
	opened := make(map[artifact.ID]int)
	byteBounded := observationStreamOverrideReader{Reader: store, opened: opened}
	if _, found, err := LoadObservationStream(ctx, byteBounded, scope.Attempt, ObservationStreamBounds{
		MaxChunks: len(wantSummaryIDs), MaxRawBytes: wantBytes - 1,
	}); err == nil || !found {
		t.Fatalf("raw byte bound refusal = (%v, %v)", found, err)
	}
	if opened[secondSummary.Chunk] != 1 || opened[firstSummary.Chunk] != 0 {
		t.Fatalf("raw opens before byte refusal = %+v", opened)
	}
	zeroOpened := make(map[artifact.ID]int)
	zeroBounded := observationStreamOverrideReader{Reader: store, opened: zeroOpened}
	if _, found, err := LoadObservationStream(ctx, zeroBounded, scope.Attempt, ObservationStreamBounds{}); err == nil || !found {
		t.Fatalf("zero bound acceptance = (%v, %v)", found, err)
	}
	if zeroOpened[firstSummary.Chunk] != 0 || zeroOpened[secondSummary.Chunk] != 0 {
		t.Fatalf("raw opens with exhausted bound = %+v", zeroOpened)
	}
	if _, found, err := LoadObservationStream(
		ctx, zeroBounded, id(artifact.KindRun, "missing-zero-bound-stream"), ObservationStreamBounds{},
	); err != nil || found {
		t.Fatalf("missing stream with exhausted bound = (%v, %v)", found, err)
	}

	secondContent, err := second.Content()
	if err != nil {
		t.Fatal(err)
	}
	secondContent.Data[0] ^= byte(traceSequenceStart)
	corrupt := observationStreamOverrideReader{
		Reader: store, contents: map[artifact.ID]artifact.Content{second.ID: secondContent},
	}
	if _, found, err := LoadObservationStream(ctx, corrupt, scope.Attempt, bounds); err == nil || !found {
		t.Fatalf("raw mismatch refusal = (%v, %v)", found, err)
	}

	refusals := []struct {
		name  string
		scope ResourceScope
		first uint32
	}{
		{name: "scope", scope: ResourceScope{
			Surface: scope.Surface, Model: scope.Model,
			Hardware: id(artifact.KindEvidence, "changed-stream-hardware"), Provider: scope.Provider,
			Workload: scope.Workload, Attempt: scope.Attempt,
		}, first: second.Samples[0].Ordinal},
		{name: "ordinal", scope: scope, first: second.Samples[0].Ordinal + traceSequenceStart},
	}
	for _, refusal := range refusals {
		t.Run("refuses "+refusal.name+" discontinuity", func(t *testing.T) {
			forgedChunk, err := NewObservationChunk(refusal.scope, firstSummary.ID, []ObservationSample{{
				Ordinal: refusal.first, ElapsedNS: second.Samples[0].ElapsedNS,
				Kind:     ObservationSampleExecution,
				Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}},
			}})
			if err != nil {
				t.Fatal(err)
			}
			forgedSummary, err := SummarizeObservationChunk(forgedChunk)
			if err != nil {
				t.Fatal(err)
			}
			chunkContent, err := forgedChunk.Content()
			if err != nil {
				t.Fatal(err)
			}
			summaryContent, err := forgedSummary.Content()
			if err != nil {
				t.Fatal(err)
			}
			reader := observationStreamOverrideReader{
				Reader: store,
				aliases: map[string]artifact.ID{
					ObservationChunkAlias(scope.Attempt): forgedSummary.ID,
				},
				contents: map[artifact.ID]artifact.Content{
					forgedChunk.ID:   chunkContent,
					forgedSummary.ID: summaryContent,
				},
			}
			if _, found, err := LoadObservationStream(ctx, reader, scope.Attempt, bounds); err == nil || !found {
				t.Fatalf("discontinuity refusal = (%v, %v)", found, err)
			}
		})
	}

	third, err := NewObservationChunk(scope, secondSummary.ID, []ObservationSample{{
		Ordinal: 4, ElapsedNS: 40, Kind: ObservationSampleExecution,
		Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 100}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := commitObservationChunk(t, ctx, store, third); err != nil {
		t.Fatal(err)
	}
	historical, err := RequireObservationStream(ctx, store, scope.Attempt, secondSummary.ID, bounds)
	if err != nil || !reflect.DeepEqual(historical, stream) {
		t.Fatalf("exact historical stream after alias advance = (%+v, %v)", historical, err)
	}

	hardwareScope := scope
	hardwareScope.Attempt = id(artifact.KindRun, "hardware-stream-attempt")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "observation/stream/hardware-attempt",
		Artifacts: []artifact.Descriptor{{ID: hardwareScope.Attempt}},
	}); err != nil {
		t.Fatal(err)
	}
	interactionsOnly := InteractionWork{ToolCalls: 1}
	hardwareChunk, err := NewObservationChunk(hardwareScope, artifact.ID{}, []ObservationSample{
		{
			Ordinal: traceSequenceStart, Kind: ObservationSampleHardware,
			Interactions: &interactionsOnly,
		},
		{
			Ordinal: traceSequenceStart + 1, Kind: ObservationSampleHardware,
			Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}},
		},
		{
			Ordinal: traceSequenceStart + 2, Kind: ObservationSampleHardware,
			Measures: []ResourceMeasure{{Metric: ResourceCPUNS, Value: 0}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	hardwareSummary, err := commitObservationChunk(t, ctx, store, hardwareChunk)
	if err != nil {
		t.Fatal(err)
	}
	hardwareStream, found, err := LoadObservationStream(ctx, store, hardwareScope.Attempt, ObservationStreamBounds{
		MaxChunks: 1, MaxRawBytes: hardwareSummary.Stats.Bytes,
	})
	if err != nil || !found {
		t.Fatalf("hardware stream = (%+v, %v, %v)", hardwareStream, found, err)
	}
	if hardwareStream.Coverage.HardwareMeasuredSamples != 1 ||
		hardwareStream.Coverage.MeasuredSamples != 2 ||
		hardwareStream.Coverage.InteractionSamples != 1 ||
		!slices.Equal(hardwareStream.Coverage.HardwareMetrics, []ObservationMetricSamples{{
			Metric: ResourceCPUNS, Samples: 1,
		}}) {
		t.Fatalf("hardware coverage = %+v", hardwareStream.Coverage)
	}
}

func scopeAuthorities(scope ResourceScope) []artifact.ID {
	return ResourceFitness{Scope: scope}.Authorities()
}
