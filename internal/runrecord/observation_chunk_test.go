package runrecord

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func commitObservationChunk(
	t *testing.T,
	ctx context.Context,
	repository artifact.Repository,
	value ObservationChunk,
) (ObservationChunkSummary, error) {
	t.Helper()
	summary, err := SummarizeObservationChunk(value)
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	batch := artifact.Batch{Key: fmt.Sprintf(
		"observation/chunk/%s/%d", value.Scope.Attempt, summary.Stats.FirstOrdinal,
	)}
	bound, err := BindObservationChunk(ctx, repository, &batch, value)
	if err != nil {
		return ObservationChunkSummary{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return ObservationChunkSummary{}, err
	}
	return bound, nil
}

func resolveObservationChunkSummary(
	ctx context.Context,
	reader artifact.Reader,
	attempt artifact.ID,
) (ObservationChunkSummary, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, ObservationChunkAlias(attempt))
	if err != nil || !found {
		return ObservationChunkSummary{}, found, err
	}
	summary, err := RequireObservationChunkSummary(ctx, reader, id)
	return summary, true, err
}

func TestObservationChunkPublication(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "source"))
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
		Model:    id(artifact.KindModel, "observation-model"),
		Hardware: id(artifact.KindEvidence, "observation-hardware"),
		Provider: id(artifact.KindProfile, "observation-provider"),
		Workload: id(artifact.KindRecipe, "observation-workload"),
		Attempt:  id(artifact.KindRun, "observation-attempt"),
	}
	authorities := []artifact.ID{scope.Model, scope.Hardware, scope.Provider, scope.Workload, scope.Attempt}
	descriptors := make([]artifact.Descriptor, len(authorities))
	for index, authority := range authorities {
		descriptors[index] = artifact.Descriptor{ID: authority}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "observation/test/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}

	observedZero := InteractionWork{}
	observedWork := InteractionWork{Commits: 1, Retries: 2}
	first, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{
		{Ordinal: 1, ElapsedNS: 10, Kind: ObservationSampleToken, Measures: []ResourceMeasure{
			{Metric: ResourcePeakHostBytes, Value: 100},
			{Metric: ResourceCostUnits, Value: 0},
			{Metric: ResourceInputTokens, Value: 3},
			{Metric: ResourceWallNS, Value: 5},
		}},
		{Ordinal: 2, ElapsedNS: 20, Kind: ObservationSampleHardware, Measures: []ResourceMeasure{
			{Metric: ResourcePeakDeviceBytes, Value: 40},
			{Metric: ResourcePeakHostBytes, Value: 80},
			{Metric: ResourceWallNS, Value: 7},
		}, Interactions: &observedZero},
		{Ordinal: 3, ElapsedNS: 30, Kind: ObservationSampleExecution, Measures: []ResourceMeasure{
			{Metric: ResourcePeakDeviceBytes, Value: 50},
			{Metric: ResourceWallNS, Value: 11},
		}, Interactions: &observedWork},
		{Ordinal: 4, ElapsedNS: 40, Kind: ObservationSampleMessage, Measures: []ResourceMeasure{
			{Metric: ResourceOutputTokens, Value: 7},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := first.Content()
	if err != nil {
		t.Fatal(err)
	}
	expected, err := SummarizeObservationChunk(first)
	if err != nil {
		t.Fatal(err)
	}
	_, sequenceBefore := store.Head()
	firstSummary, err := commitObservationChunk(t, ctx, store, first)
	if err != nil {
		t.Fatal(err)
	}
	_, sequenceAfter := store.Head()
	if sequenceAfter != sequenceBefore+1 || !reflect.DeepEqual(firstSummary, expected) {
		t.Fatalf("atomic publication = sequence %d->%d summary=%+v", sequenceBefore, sequenceAfter, firstSummary)
	}

	storedRaw, found, err := artifact.ReadContent(ctx, store, firstSummary.Chunk)
	if err != nil || !found {
		t.Fatalf("raw content = (%+v, %v, %v)", storedRaw.Descriptor, found, err)
	}
	storedSummary, found, err := artifact.ReadContent(ctx, store, firstSummary.ID)
	if err != nil || !found {
		t.Fatalf("summary content = (%+v, %v, %v)", storedSummary.Descriptor, found, err)
	}
	if storedRaw.Descriptor.Schema != "" || storedRaw.Descriptor.MediaType != ObservationChunkMediaType ||
		overgodb.DocumentStorageClass(storedRaw.Descriptor) != overgodb.ClassImmutableBlob {
		t.Fatalf("raw storage contract = %+v class=%s", storedRaw.Descriptor, overgodb.DocumentStorageClass(storedRaw.Descriptor))
	}
	if storedSummary.Descriptor.Schema != ObservationChunkSummarySchema || storedSummary.Descriptor.MediaType != ObservationChunkSummaryMediaType ||
		overgodb.DocumentStorageClass(storedSummary.Descriptor) != overgodb.ClassCanonicalFact {
		t.Fatalf("summary storage contract = %+v class=%s", storedSummary.Descriptor, overgodb.DocumentStorageClass(storedSummary.Descriptor))
	}
	digest, err := artifact.IdentifyBytes(artifact.KindFile, storedRaw.Data)
	if err != nil || digest != firstSummary.Chunk || firstSummary.Chunk != raw.Descriptor.ID || firstSummary.Stats.Bytes != uint64(len(storedRaw.Data)) {
		t.Fatalf("raw digest/bytes = (%s, %s, %d, %v)", digest, firstSummary.Chunk, firstSummary.Stats.Bytes, err)
	}
	parsed, err := ParseObservationChunk(storedRaw.Data)
	if err != nil || !reflect.DeepEqual(parsed, first) {
		t.Fatalf("raw round trip = (%+v, %v)", parsed, err)
	}
	verifiedSummary, err := RequireObservationChunkSummary(ctx, store, firstSummary.ID)
	if err != nil || !reflect.DeepEqual(verifiedSummary, firstSummary) {
		t.Fatalf("summary round trip = (%+v, %v)", verifiedSummary, err)
	}
	if parsed.Samples[0].Interactions != nil || parsed.Samples[1].Interactions == nil || *parsed.Samples[1].Interactions != (InteractionWork{}) {
		t.Fatalf("unknown and observed-zero interactions collapsed: %+v", parsed.Samples)
	}

	wantMetrics := []ObservationMetricSamples{
		{Metric: ResourceCostUnits, Samples: 1},
		{Metric: ResourceInputTokens, Samples: 1},
		{Metric: ResourceOutputTokens, Samples: 1},
		{Metric: ResourcePeakDeviceBytes, Samples: 2},
		{Metric: ResourcePeakHostBytes, Samples: 2},
		{Metric: ResourceWallNS, Samples: 3},
	}
	wantKinds := []ObservationKindSamples{
		{Kind: ObservationSampleExecution, Samples: 1},
		{Kind: ObservationSampleHardware, Samples: 1},
		{Kind: ObservationSampleMessage, Samples: 1},
		{Kind: ObservationSampleToken, Samples: 1},
	}
	if firstSummary.Stats.Samples != 4 || firstSummary.Stats.FirstOrdinal != 1 || firstSummary.Stats.LastOrdinal != 4 ||
		firstSummary.Stats.FirstElapsedNS != 10 || firstSummary.Stats.LastElapsedNS != 40 ||
		firstSummary.Stats.InteractionSamples != 2 || !slices.Equal(firstSummary.Stats.Metrics, wantMetrics) ||
		!slices.Equal(firstSummary.Stats.Kinds, wantKinds) {
		t.Fatalf("summary stats = %+v", firstSummary.Stats)
	}
	for metric, want := range map[ResourceMetric]uint64{
		ResourceCostUnits: 0, ResourceInputTokens: 3, ResourceOutputTokens: 7,
		ResourcePeakDeviceBytes: 50, ResourcePeakHostBytes: 100, ResourceWallNS: 23,
	} {
		if got, known := firstSummary.Aggregate.Measure(metric); !known || got != want {
			t.Fatalf("aggregate %s = (%d, %v), want (%d, true)", metric, got, known, want)
		}
	}
	if _, known := firstSummary.Aggregate.Measure(ResourceGPUNS); known {
		t.Fatal("unknown GPU time became an observed zero")
	}
	if firstSummary.Aggregate.Interactions == nil || *firstSummary.Aggregate.Interactions != observedWork {
		t.Fatalf("interaction aggregate = %+v", firstSummary.Aggregate.Interactions)
	}
	parents, err := store.Parents(ctx, firstSummary.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(parents, artifact.Lineage{Child: firstSummary.ID, Parent: firstSummary.Chunk, Relation: artifact.RelationContains}) {
		t.Fatalf("summary lacks raw digest lineage: %+v", parents)
	}
	for _, authority := range authorities {
		if !slices.Contains(parents, artifact.Lineage{Child: firstSummary.ID, Parent: authority, Relation: artifact.RelationDependsOn}) {
			t.Fatalf("summary lacks source %s: %+v", authority, parents)
		}
	}
	current, found, err := resolveObservationChunkSummary(ctx, store, scope.Attempt)
	if err != nil || !found || current.ID != firstSummary.ID {
		t.Fatalf("first stream head = (%+v, %v, %v)", current, found, err)
	}

	noncanonical := append([]byte{' '}, storedRaw.Data...)
	if _, err := ParseObservationChunk(noncanonical); err == nil {
		t.Fatal("non-canonical raw bytes accepted")
	}

	divergent, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{{
		Ordinal: 1, ElapsedNS: 10, Kind: ObservationSampleToken,
		Measures: []ResourceMeasure{{Metric: ResourceInputTokens, Value: 4}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	divergentSummary, err := SummarizeObservationChunk(divergent)
	if err != nil {
		t.Fatal(err)
	}
	headBefore, sequenceBefore := store.Head()
	if _, err := commitObservationChunk(t, ctx, store, divergent); err == nil {
		t.Fatal("divergent initial range accepted")
	}
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		t.Fatalf("divergent refusal advanced head = (%s, %d)", head, sequence)
	}
	if _, found, err := store.Artifact(ctx, divergentSummary.ID); err != nil || found {
		t.Fatalf("divergent summary became visible = (%v, %v)", found, err)
	}

	second, err := NewObservationChunk(scope, firstSummary.ID, []ObservationSample{{
		Ordinal: 5, ElapsedNS: 45, Kind: ObservationSampleExecution,
		Measures: []ResourceMeasure{{Metric: ResourceDiskReadBytes, Value: 9}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	staleSibling, err := NewObservationChunk(scope, firstSummary.ID, []ObservationSample{{
		Ordinal: 5, ElapsedNS: 46, Kind: ObservationSampleExecution,
		Measures: []ResourceMeasure{{Metric: ResourceDiskReadBytes, Value: 10}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	staleSummary, err := SummarizeObservationChunk(staleSibling)
	if err != nil {
		t.Fatal(err)
	}
	_, sequenceBefore = store.Head()
	secondSummary, err := commitObservationChunk(t, ctx, store, second)
	if err != nil {
		t.Fatal(err)
	}
	_, sequenceAfter = store.Head()
	if sequenceAfter != sequenceBefore+1 || secondSummary.Previous != firstSummary.ID || secondSummary.Stats.FirstOrdinal != 5 {
		t.Fatalf("continuation = sequence %d->%d summary=%+v", sequenceBefore, sequenceAfter, secondSummary)
	}
	headBefore, sequenceBefore = store.Head()
	if _, err := commitObservationChunk(t, ctx, store, staleSibling); err == nil {
		t.Fatal("stale sibling accepted")
	}
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		t.Fatalf("stale refusal advanced head = (%s, %d)", head, sequence)
	}
	if _, found, err := store.Artifact(ctx, staleSummary.ID); err != nil || found {
		t.Fatalf("stale summary became visible = (%v, %v)", found, err)
	}

	headBefore, sequenceBefore = store.Head()
	replayed, err := commitObservationChunk(t, ctx, store, first)
	if err != nil || !reflect.DeepEqual(replayed, firstSummary) {
		t.Fatalf("historical exact replay = (%+v, %v)", replayed, err)
	}
	if head, sequence := store.Head(); head != headBefore || sequence != sequenceBefore {
		t.Fatalf("exact replay advanced head = (%s, %d)", head, sequence)
	}
	current, found, err = resolveObservationChunkSummary(ctx, store, scope.Attempt)
	if err != nil || !found || current.ID != secondSummary.ID {
		t.Fatalf("continuation stream head = (%+v, %v, %v)", current, found, err)
	}

	other, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{{
		Ordinal: 1, ElapsedNS: 10, Kind: ObservationSampleToken,
		Measures: []ResourceMeasure{{Metric: ResourceInputTokens, Value: 999}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyObservationChunk(firstSummary, other); err == nil {
		t.Fatal("summary verified a different raw chunk")
	}
	otherContent, err := other.Content()
	if err != nil {
		t.Fatal(err)
	}
	forgedBody := cloneObservationChunkSummary(firstSummary)
	forgedBody.Chunk, forgedBody.ID = other.ID, artifact.ID{}
	forged, err := observationChunkSummaryCodec.New(forgedBody)
	if err != nil {
		t.Fatal(err)
	}
	forgedContent, err := forged.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "observation/test/forged-summary", Contents: []artifact.Content{otherContent, forgedContent},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireObservationChunkSummary(ctx, store, forged.ID); err == nil {
		t.Fatal("stored summary/raw mismatch verified")
	}

	compactedRoot := filepath.Join(root, "compacted")
	if _, err := overgodb.Compact(ctx, store, compactedRoot); err != nil {
		t.Fatal(err)
	}
	compacted, err := overgodb.OpenReadOnly(compactedRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	for _, summary := range []ObservationChunkSummary{firstSummary, secondSummary} {
		if _, err := RequireObservationChunkSummary(ctx, compacted, summary.ID); err != nil {
			t.Fatalf("compaction lost summary %s: %v", summary.ID, err)
		}
		if _, err := RequireObservationChunk(ctx, compacted, summary.Chunk); err != nil {
			t.Fatalf("compaction lost raw chunk %s: %v", summary.Chunk, err)
		}
	}
	current, found, err = resolveObservationChunkSummary(ctx, compacted, scope.Attempt)
	if err != nil || !found || current.ID != secondSummary.ID {
		t.Fatalf("compacted stream head = (%+v, %v, %v)", current, found, err)
	}
	if _, found, err := compacted.Artifact(ctx, forged.ID); err != nil || found {
		t.Fatalf("unrooted forged summary retained = (%v, %v)", found, err)
	}

	refusalSample := func(ordinal uint32, elapsed uint64) ObservationSample {
		return ObservationSample{
			Ordinal: ordinal, ElapsedNS: elapsed, Kind: ObservationSampleToken,
			Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}},
		}
	}
	foreignSummary := id(artifact.KindEvidence, "foreign-observation-summary")
	refusals := []struct {
		name     string
		scope    ResourceScope
		previous artifact.ID
		samples  []ObservationSample
	}{
		{name: "empty count", scope: scope},
		{name: "empty sample", scope: scope, samples: []ObservationSample{{Ordinal: 1, Kind: ObservationSampleToken}}},
		{name: "sequence gap", scope: scope, samples: []ObservationSample{refusalSample(1, 1), refusalSample(3, 2)}},
		{name: "elapsed regression", scope: scope, samples: []ObservationSample{refusalSample(1, 2), refusalSample(2, 1)}},
		{name: "foreign kind", scope: scope, samples: []ObservationSample{{
			Ordinal: 1, Kind: "foreign", Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}},
		}}},
		{name: "foreign metric", scope: scope, samples: []ObservationSample{{
			Ordinal: 1, Kind: ObservationSampleToken, Measures: []ResourceMeasure{{Metric: "joules", Value: 1}},
		}}},
		{name: "duplicate metric", scope: scope, samples: []ObservationSample{{
			Ordinal: 1, Kind: ObservationSampleToken,
			Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}, {Metric: ResourceWallNS, Value: 2}},
		}}},
		{name: "foreign scope", scope: ResourceScope{
			Surface: "foreign", Model: scope.Model, Hardware: scope.Hardware, Provider: scope.Provider,
			Workload: scope.Workload, Attempt: scope.Attempt,
		}, samples: []ObservationSample{refusalSample(1, 1)}},
		{name: "initial has previous", scope: scope, previous: foreignSummary, samples: []ObservationSample{refusalSample(1, 1)}},
		{name: "continuation lacks previous", scope: scope, samples: []ObservationSample{refusalSample(2, 1)}},
	}
	for _, refusal := range refusals {
		t.Run("refuses "+refusal.name, func(t *testing.T) {
			if _, err := NewObservationChunk(refusal.scope, refusal.previous, refusal.samples); err == nil {
				t.Fatal("accepted")
			}
		})
	}
	if _, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{
		{Ordinal: 1, Kind: ObservationSampleToken, Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: math.MaxUint64}}},
		{Ordinal: 2, Kind: ObservationSampleToken, Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 1}}},
	}); err == nil {
		t.Fatal("resource aggregate overflow accepted")
	}
	maxWork, oneWork := InteractionWork{Retries: math.MaxUint64}, InteractionWork{Retries: 1}
	if _, err := NewObservationChunk(scope, artifact.ID{}, []ObservationSample{
		{Ordinal: 1, Kind: ObservationSampleExecution, Interactions: &maxWork},
		{Ordinal: 2, Kind: ObservationSampleExecution, Interactions: &oneWork},
	}); err == nil {
		t.Fatal("interaction aggregate overflow accepted")
	}

	func() {
		tooMany := make([]ObservationSample, ObservationChunkMaximumSamples+1)
		if _, err := NewObservationChunk(scope, artifact.ID{}, tooMany); err == nil {
			t.Fatal("sample count above bound accepted")
		}
	}()
	runtime.GC()
	func() {
		oversized := make([]byte, ObservationChunkMaximumBytes+1)
		if _, err := ParseObservationChunk(oversized); err == nil {
			t.Fatal("raw bytes above bound accepted")
		}
	}()
}
