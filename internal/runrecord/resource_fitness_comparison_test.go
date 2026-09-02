package runrecord

import (
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestImprovementFitnessRejectsTransferredCost(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	id := func(kind artifact.Kind, label string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, "fitness comparison "+label)
	}
	hardware := id(artifact.KindEvidence, "hardware")
	providerImplementation := id(artifact.KindFile, "provider implementation")
	providerSchema := id(artifact.KindProfile, "provider schema")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fitness-comparison/provider-parents",
		Artifacts: []artifact.Descriptor{
			{ID: providerImplementation},
			{ID: providerSchema},
		},
	}); err != nil {
		t.Fatal(err)
	}
	providerIdentity, err := (CapabilityIdentity{
		Implementation: providerImplementation,
		Release:        "1.0.0",
		Transport: CapabilityTransport{
			Kind: CapabilityTransportBuiltin, Protocol: "cost-units/1",
		},
		Schema:   providerSchema,
		Platform: CapabilityPlatform{OS: "test", Arch: "test"},
		Resources: CapabilityResourceEnvelope{
			MaxInputBytes: 1, MaxOutputBytes: 1, MaxConcurrent: 1, CPUThreads: 1, HostBytes: 1,
		},
	}).Identify()
	if err != nil {
		t.Fatal(err)
	}
	providerContent, err := providerIdentity.Content()
	if err != nil {
		t.Fatal(err)
	}
	providerBatch, err := artifact.NewDocumentBatch(
		"fitness-comparison/provider", []artifact.Content{providerContent}, providerIdentity.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, providerBatch); err != nil {
		t.Fatal(err)
	}
	provider := providerIdentity.ID
	baselineModel := id(artifact.KindModel, "baseline model")
	candidateModel := id(artifact.KindModel, "candidate model")
	newScope := func(surface InteractionSurface, role, arm string) ResourceScope {
		return ResourceScope{
			Surface: surface, Model: map[string]artifact.ID{
				"baseline": baselineModel, "candidate": candidateModel,
			}[arm],
			Hardware: hardware, Provider: provider,
			Workload: id(artifact.KindProfile, role+" "+arm+" workload"),
			Attempt:  id(artifact.KindEvidence, role+" "+arm+" attempt"),
		}
	}
	agentBaselineScope := newScope(SurfaceAgent, ResourceLaneAgent, "baseline")
	agentCandidateScope := newScope(SurfaceAgent, ResourceLaneAgent, "candidate")
	toolBaselineScope := newScope(SurfaceTool, ResourceLaneTool, "baseline")
	toolCandidateScope := newScope(SurfaceTool, ResourceLaneTool, "candidate")
	var authorities []artifact.ID
	for _, scope := range []ResourceScope{
		agentBaselineScope, agentCandidateScope, toolBaselineScope, toolCandidateScope,
	} {
		authorities = append(authorities, (ResourceFitness{Scope: scope}).Authorities()...)
	}
	slices.SortFunc(authorities, artifact.CompareID)
	authorities = slices.Compact(authorities)
	descriptors := make([]artifact.Descriptor, 0, len(authorities))
	for _, authority := range authorities {
		if authority != provider {
			descriptors = append(descriptors, artifact.Descriptor{ID: authority})
		}
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fitness-comparison/authorities", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}

	loadChunk := func(chunk ObservationChunk) ObservationStream {
		t.Helper()
		summary, err := commitObservationChunk(t, ctx, store, chunk)
		if err != nil {
			t.Fatal(err)
		}
		loaded, err := RequireObservationStream(ctx, store, chunk.Scope.Attempt, summary.ID, ObservationStreamBounds{
			MaxChunks: 1, MaxRawBytes: summary.Stats.Bytes,
		})
		if err != nil {
			t.Fatal(err)
		}
		return loaded
	}
	stream := func(scope ResourceScope, cost, wall, peak, failures, retries uint64) ObservationStream {
		t.Helper()
		work := InteractionWork{Failures: failures, Retries: retries, ToolCalls: 1, SemanticTransitions: 2}
		fitness, err := NewResourceFitness(ResourceFitness{
			Scope: scope,
			Measures: []ResourceMeasure{
				{Metric: ResourceCostUnits, Value: cost},
				{Metric: ResourcePeakHostBytes, Value: peak},
				{Metric: ResourceWallNS, Value: wall},
			},
			Interactions: &work,
		})
		if err != nil {
			t.Fatal(err)
		}
		chunk, err := NewInitialObservationChunk(fitness, ObservationSampleExecution, wall)
		if err != nil {
			t.Fatal(err)
		}
		return loadChunk(chunk)
	}
	agentBaseline := stream(agentBaselineScope, 10, 100, 1_000, 1, 2)
	agentCandidate := stream(agentCandidateScope, 5, 90, 900, 0, 1)
	toolBaseline := stream(toolBaselineScope, 1, 10, 100, 0, 0)
	toolTransferred := stream(toolCandidateScope, 7, 10, 100, 0, 0)
	required := []ResourceMetric{ResourceCostUnits, ResourcePeakHostBytes, ResourceWallNS}
	transferred := []ResourceFitnessLane{
		{
			Name: ResourceLaneAgent, RequiredMetrics: required, RequireInteractions: true,
			Baseline: agentBaseline, Candidate: agentCandidate,
		},
		{
			Name: ResourceLaneTool, RequiredMetrics: required, RequireInteractions: true,
			Baseline: toolBaseline, Candidate: toolTransferred,
		},
	}
	if _, err := CompareResourceFitness(ctx, store, transferred); err == nil {
		t.Fatal("candidate transferred cost from agent lane into tool lane")
	}
	toolFailureScope := toolCandidateScope
	toolFailureScope.Attempt = id(artifact.KindEvidence, "tool failure candidate attempt")
	testutil.PublishArtifact(t, store, toolFailureScope.Attempt)
	toolFailureTransferred := stream(toolFailureScope, 1, 10, 100, 1, 0)
	if _, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{
		transferred[0], {
			Name: ResourceLaneTool, RequiredMetrics: required, RequireInteractions: true,
			Baseline: toolBaseline, Candidate: toolFailureTransferred,
		},
	}); err == nil {
		t.Fatal("candidate transferred a failure from agent lane into tool lane")
	}

	missingProviderBaselineScope := agentBaselineScope
	missingProviderBaselineScope.Provider = artifact.ID{}
	missingProviderBaselineScope.Attempt = id(artifact.KindEvidence, "missing provider baseline attempt")
	missingProviderCandidateScope := agentCandidateScope
	missingProviderCandidateScope.Provider = artifact.ID{}
	missingProviderCandidateScope.Attempt = id(artifact.KindEvidence, "missing provider candidate attempt")
	for _, attempt := range []artifact.ID{missingProviderBaselineScope.Attempt, missingProviderCandidateScope.Attempt} {
		testutil.PublishArtifact(t, store, attempt)
	}
	missingProviderBaseline := stream(missingProviderBaselineScope, 1, 10, 100, 0, 0)
	missingProviderCandidate := stream(missingProviderCandidateScope, 1, 10, 100, 0, 0)
	if _, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{{
		Name: ResourceLaneAgent, RequiredMetrics: []ResourceMetric{ResourcePeakHostBytes, ResourceWallNS},
		Baseline: missingProviderBaseline, Candidate: missingProviderCandidate,
	}}); err == nil {
		t.Fatal("cost observations admitted an absent provider")
	}

	untypedProvider := id(artifact.KindProfile, "untyped provider")
	testutil.PublishArtifact(t, store, untypedProvider)
	untypedProviderBaselineScope := agentBaselineScope
	untypedProviderBaselineScope.Provider = untypedProvider
	untypedProviderBaselineScope.Attempt = id(artifact.KindEvidence, "untyped provider baseline attempt")
	untypedProviderCandidateScope := agentCandidateScope
	untypedProviderCandidateScope.Provider = untypedProvider
	untypedProviderCandidateScope.Attempt = id(artifact.KindEvidence, "untyped provider candidate attempt")
	for _, attempt := range []artifact.ID{untypedProviderBaselineScope.Attempt, untypedProviderCandidateScope.Attempt} {
		testutil.PublishArtifact(t, store, attempt)
	}
	untypedProviderBaseline := stream(untypedProviderBaselineScope, 1, 10, 100, 0, 0)
	untypedProviderCandidate := stream(untypedProviderCandidateScope, 1, 10, 100, 0, 0)
	if _, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{{
		Name: ResourceLaneAgent, RequiredMetrics: required,
		Baseline: untypedProviderBaseline, Candidate: untypedProviderCandidate,
	}}); err == nil {
		t.Fatal("cost observations admitted a descriptor-only provider")
	}

	// A candidate cannot manufacture an improvement by omitting observations.
	// These streams are genuine source-backed aggregates: without equal
	// denominators, the shorter candidate hides the baseline's final failure,
	// retry, and resource work while appearing strictly cheaper.
	omittedBaselineScope := agentBaselineScope
	omittedBaselineScope.Attempt = id(artifact.KindEvidence, "omitted sample baseline attempt")
	omittedCandidateScope := agentCandidateScope
	omittedCandidateScope.Attempt = id(artifact.KindEvidence, "omitted sample candidate attempt")
	hiddenWorkCandidateScope := agentCandidateScope
	hiddenWorkCandidateScope.Attempt = id(artifact.KindEvidence, "hidden work candidate attempt")
	for _, attempt := range []artifact.ID{
		omittedBaselineScope.Attempt, omittedCandidateScope.Attempt, hiddenWorkCandidateScope.Attempt,
	} {
		testutil.PublishArtifact(t, store, attempt)
	}
	observedWork := InteractionWork{ToolCalls: 1, SemanticTransitions: 2}
	hiddenWork := InteractionWork{Failures: 1, Retries: 1}
	completeChunk, err := NewObservationChunk(omittedBaselineScope, artifact.ID{}, []ObservationSample{
		{
			Ordinal: 1, ElapsedNS: 50, Kind: ObservationSampleExecution,
			Measures: []ResourceMeasure{
				{Metric: ResourceCostUnits, Value: 5},
				{Metric: ResourcePeakHostBytes, Value: 1_000},
				{Metric: ResourceWallNS, Value: 50},
			},
			Interactions: &observedWork,
		},
		{
			Ordinal: 2, ElapsedNS: 100, Kind: ObservationSampleMessage,
			Measures: []ResourceMeasure{
				{Metric: ResourceCostUnits, Value: 5},
				{Metric: ResourcePeakHostBytes, Value: 1_000},
				{Metric: ResourceWallNS, Value: 50},
			},
			Interactions: &hiddenWork,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	completeBaseline := loadChunk(completeChunk)
	omittedChunk, err := NewObservationChunk(omittedCandidateScope, artifact.ID{}, []ObservationSample{{
		Ordinal: 1, ElapsedNS: 90, Kind: ObservationSampleExecution,
		Measures: []ResourceMeasure{
			{Metric: ResourceCostUnits, Value: 5},
			{Metric: ResourcePeakHostBytes, Value: 900},
			{Metric: ResourceWallNS, Value: 90},
		},
		Interactions: &observedWork,
	}})
	if err != nil {
		t.Fatal(err)
	}
	omittedCandidate := loadChunk(omittedChunk)
	if _, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{{
		Name: ResourceLaneAgent, RequiredMetrics: required, RequireInteractions: true,
		Baseline: completeBaseline, Candidate: omittedCandidate,
	}}); err == nil {
		t.Fatal("candidate hid failure and resource work by providing fewer samples")
	}

	// Equal total, metric, and sample-kind coverage is still insufficient when
	// a candidate drops the interaction observation that carried failed work.
	hiddenWorkChunk, err := NewObservationChunk(hiddenWorkCandidateScope, artifact.ID{}, []ObservationSample{
		{
			Ordinal: 1, ElapsedNS: 45, Kind: ObservationSampleExecution,
			Measures: []ResourceMeasure{
				{Metric: ResourceCostUnits, Value: 5},
				{Metric: ResourcePeakHostBytes, Value: 900},
				{Metric: ResourceWallNS, Value: 90},
			},
			Interactions: &observedWork,
		},
		{
			Ordinal: 2, ElapsedNS: 90, Kind: ObservationSampleMessage,
			Measures: []ResourceMeasure{
				{Metric: ResourceCostUnits, Value: 0},
				{Metric: ResourcePeakHostBytes, Value: 900},
				{Metric: ResourceWallNS, Value: 0},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	hiddenWorkCandidate := loadChunk(hiddenWorkChunk)
	if _, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{{
		Name: ResourceLaneAgent, RequiredMetrics: required, RequireInteractions: true,
		Baseline: completeBaseline, Candidate: hiddenWorkCandidate,
	}}); err == nil {
		t.Fatal("candidate hid failure and retry work by omitting an interaction observation")
	}

	toolCandidate := toolTransferred
	for index := range toolCandidate.Aggregate.Measures {
		if toolCandidate.Aggregate.Measures[index].Metric == ResourceCostUnits {
			toolCandidate.Aggregate.Measures[index].Value = 1
		}
	}
	// The adjusted value is deliberately structural first: exact source replay
	// must reject it even though its no-regression arithmetic is valid.
	structural, err := NewResourceFitnessComparison([]ResourceFitnessLane{
		transferred[0], {
			Name: ResourceLaneTool, RequiredMetrics: required, RequireInteractions: true,
			Baseline: toolBaseline, Candidate: toolCandidate,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := structural.VerifiedBatch(ctx, store, "fitness-comparison/forged"); err == nil {
		t.Fatal("verified publication accepted a forged aggregate")
	}
	forgedBatch, err := resourceFitnessComparisonCodec.Batch(
		"fitness-comparison/forged-bypass", structural, structural.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, forgedBatch); err != nil {
		t.Fatal(err)
	}
	rejected, err := RequireResourceFitnessComparison(ctx, store, structural.ID)
	if err == nil || rejected.ID.Valid() || rejected.Lanes != nil {
		t.Fatalf("read-side verification accepted forged aggregate: %+v err=%v", rejected, err)
	}

	// Publish a genuine equal-cost tool stream and compare both complete lanes.
	genuineToolCandidateScope := toolCandidateScope
	genuineToolCandidateScope.Attempt = id(artifact.KindEvidence, "tool genuine candidate attempt")
	testutil.PublishArtifact(t, store, genuineToolCandidateScope.Attempt)
	genuineToolCandidate := stream(genuineToolCandidateScope, 1, 10, 100, 0, 0)
	lanes := []ResourceFitnessLane{
		transferred[0], {
			Name: ResourceLaneTool, RequiredMetrics: required, RequireInteractions: true,
			Baseline: toolBaseline, Candidate: genuineToolCandidate,
		},
	}
	comparison, err := CompareResourceFitness(ctx, store, lanes)
	if err != nil {
		t.Fatal(err)
	}
	if !comparison.StrictImprovement || comparison.Lanes[0].Name != ResourceLaneAgent ||
		comparison.Lanes[0].StrictInteractions == nil || comparison.Lanes[0].StrictInteractions.Failures != 1 ||
		comparison.Lanes[0].StrictInteractions.Retries != 1 {
		t.Fatalf("strict improvement facts = %+v", comparison)
	}
	if got := comparison.Lanes[0].StrictMetrics; !reflect.DeepEqual(got, []ResourceMeasure{
		{Metric: ResourceCostUnits, Value: 5},
		{Metric: ResourcePeakHostBytes, Value: 100},
		{Metric: ResourceWallNS, Value: 10},
	}) {
		t.Fatalf("strict metric advantages = %+v", got)
	}
	wantChunks := []artifact.ID{
		agentBaseline.ChunkIDs[0], agentCandidate.ChunkIDs[0],
		toolBaseline.ChunkIDs[0], genuineToolCandidate.ChunkIDs[0],
	}
	slices.SortFunc(wantChunks, artifact.CompareID)
	if !slices.Equal(comparison.SourceChunks(), wantChunks) {
		t.Fatalf("source chunks = %v, want %v", comparison.SourceChunks(), wantChunks)
	}
	batch, err := comparison.VerifiedBatch(ctx, store, "fitness-comparison/genuine")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}

	// Advance an endpoint alias. The stored comparison remains bound to and
	// re-verifiable from its historical summary head.
	oldHead := agentCandidate.SummaryIDs[len(agentCandidate.SummaryIDs)-1]
	next, err := NewObservationChunk(agentCandidate.Scope, oldHead, []ObservationSample{{
		Ordinal: 2, ElapsedNS: 200, Kind: ObservationSampleExecution,
		Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: 200}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := commitObservationChunk(t, ctx, store, next); err != nil {
		t.Fatal(err)
	}
	requiredComparison, err := RequireResourceFitnessComparison(ctx, store, comparison.ID)
	if err != nil || !reflect.DeepEqual(requiredComparison, comparison) {
		t.Fatalf("historical comparison after alias advance = (%+v, %v)", requiredComparison, err)
	}

	// Repeats of the baseline set the measured noise envelope: a candidate
	// inside it is neither a regression nor a strict improvement.
	repeatScope := agentBaselineScope
	repeatScope.Attempt = id(artifact.KindEvidence, "agent repeat attempt")
	envelopeCandidateScope := agentCandidateScope
	envelopeCandidateScope.Attempt = id(artifact.KindEvidence, "agent envelope candidate attempt")
	for _, attempt := range []artifact.ID{repeatScope.Attempt, envelopeCandidateScope.Attempt} {
		testutil.PublishArtifact(t, store, attempt)
	}
	repeat := stream(repeatScope, 10, 110, 1_000, 1, 2)
	withinEnvelope := stream(envelopeCandidateScope, 10, 108, 1_000, 1, 2)
	envelopeLane := ResourceFitnessLane{
		Name: ResourceLaneAgent, RequiredMetrics: required, RequireInteractions: true,
		Baseline: agentBaseline, Repeats: []ObservationStream{repeat}, Candidate: withinEnvelope,
	}
	enveloped, err := CompareResourceFitness(ctx, store, []ResourceFitnessLane{envelopeLane})
	if err != nil {
		t.Fatal(err)
	}
	if got := enveloped.Lanes[0].NoiseEnvelope; !reflect.DeepEqual(got, []ResourceMeasure{
		{Metric: ResourceCostUnits, Value: 0},
		{Metric: ResourcePeakHostBytes, Value: 0},
		{Metric: ResourceWallNS, Value: 10},
	}) {
		t.Fatalf("noise envelope = %+v", got)
	}
	if enveloped.StrictImprovement || len(enveloped.Lanes[0].StrictMetrics) != 0 {
		t.Fatalf("a candidate inside the noise envelope counted as an improvement: %+v", enveloped.Lanes[0])
	}
	if !slices.Contains(enveloped.SourceChunks(), repeat.ChunkIDs[0]) {
		t.Fatal("the repeat's chunk is not a cited source")
	}
	if _, err := NewResourceFitnessComparison([]ResourceFitnessLane{{
		Name: ResourceLaneAgent, RequiredMetrics: required, RequireInteractions: true,
		Baseline: agentBaseline, Repeats: []ObservationStream{agentBaseline}, Candidate: withinEnvelope,
	}}); err == nil {
		t.Fatal("a repeat reusing the baseline attempt was admitted")
	}
	beyond := cloneResourceFitnessLanes([]ResourceFitnessLane{envelopeLane})
	setResourceMeasure(&beyond[0].Candidate.Aggregate, ResourceWallNS, 111)
	if _, err := NewResourceFitnessComparison(beyond); err == nil {
		t.Fatal("a candidate beyond the noise envelope was admitted")
	}
	strict := cloneResourceFitnessLanes([]ResourceFitnessLane{envelopeLane})
	setResourceMeasure(&strict[0].Candidate.Aggregate, ResourceWallNS, 85)
	strictComparison, err := NewResourceFitnessComparison(strict)
	if err != nil || !strictComparison.StrictImprovement ||
		!reflect.DeepEqual(strictComparison.Lanes[0].StrictMetrics, []ResourceMeasure{{Metric: ResourceWallNS, Value: 15}}) {
		t.Fatalf("strict improvement beyond the envelope = %+v, %v", strictComparison.Lanes[0].StrictMetrics, err)
	}

	refusals := []struct {
		name   string
		mutate func([]ResourceFitnessLane)
	}{
		{name: "latency regression", mutate: func(value []ResourceFitnessLane) {
			setResourceMeasure(&value[0].Candidate.Aggregate, ResourceWallNS, 101)
		}},
		{name: "memory regression", mutate: func(value []ResourceFitnessLane) {
			setResourceMeasure(&value[0].Candidate.Aggregate, ResourcePeakHostBytes, 1_001)
		}},
		{name: "retry regression", mutate: func(value []ResourceFitnessLane) {
			value[0].Candidate.Aggregate.Interactions.Retries = 3
		}},
		{name: "failure regression", mutate: func(value []ResourceFitnessLane) {
			value[0].Candidate.Aggregate.Interactions.Failures = 2
		}},
		{name: "missing metric", mutate: func(value []ResourceFitnessLane) {
			value[0].Candidate.Aggregate.Measures = value[0].Candidate.Aggregate.Measures[1:]
			value[0].Candidate.Coverage.Metrics = value[0].Candidate.Coverage.Metrics[1:]
		}},
		{name: "missing interactions", mutate: func(value []ResourceFitnessLane) {
			value[0].Candidate.Aggregate.Interactions = nil
			value[0].Candidate.Coverage.InteractionSamples = 0
		}},
	}
	for _, refusal := range refusals {
		t.Run(refusal.name, func(t *testing.T) {
			candidate := cloneResourceFitnessLanes(lanes)
			refusal.mutate(candidate)
			if _, err := NewResourceFitnessComparison(candidate); err == nil {
				t.Fatal("invalid comparison was admitted")
			}
		})
	}
}

func setResourceMeasure(value *ResourceFitness, metric ResourceMetric, amount uint64) {
	for index := range value.Measures {
		if value.Measures[index].Metric == metric {
			value.Measures[index].Value = amount
			return
		}
	}
}
