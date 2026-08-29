package runrecord

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const (
	servingFixtureStartedNS = int64(1_700_000_000_000_000_000)
	servingFixtureElapsedNS = uint64(25_000_000)
)

func servingObservationFixture(t *testing.T) ServingObservation {
	t.Helper()
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	return ServingObservation{
		Model: id(artifact.KindModel, "serving-model"), Recipe: id(artifact.KindRecipe, "serving-recipe"),
		Environment: id(artifact.KindEvidence, "serving-environment"),
		Operation:   id(artifact.KindEvidence, "serving-operation"), Task: recipe.TaskInference,
		Outcome: OutcomeSucceeded, StartedUnixNS: servingFixtureStartedNS, MeasuredNS: servingFixtureElapsedNS,
		SessionReused: true,
		Usage:         ServingUsage{InputTokens: 3, OutputTokens: 5, InputBytes: 13, OutputBytes: 21},
		Resources:     ServingResources{PeakHostBytes: 34, PeakDeviceBytes: 55, HostToDeviceBytes: 8, DeviceToHostBytes: 2},
		Phases:        []PhaseMetric{{Phase: PhaseDecode, DurationNS: 11}, {Phase: PhasePrefill, DurationNS: 7}},
		Hardware: []ServingHardwareSample{
			{Stage: ServingHardwareStart, DeviceCurrentBytes: 34, DevicePeakBytes: 55, DeviceAllocations: 2},
			{Stage: ServingHardwareFinish, ElapsedNS: servingFixtureElapsedNS, DeviceCurrentBytes: 21, DevicePeakBytes: 55, DeviceAllocations: 2},
		},
	}
}

func TestServingObservationContractAndIndexedQuery(t *testing.T) {
	fixture := servingObservationFixture(t)
	fixture.Version = artifact.InitialDocumentVersion
	observation, err := servingObservationCodec.New(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if err := observation.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	content, err := observation.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := servingObservationCodec.Parse(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != observation.ID || parsed.Usage != fixture.Usage || parsed.Resources != fixture.Resources ||
		!slices.Equal(parsed.Phases, []PhaseMetric{{Phase: PhaseDecode, DurationNS: 11}, {Phase: PhasePrefill, DurationNS: 7}}) ||
		!slices.Equal(parsed.Hardware, fixture.Hardware) {
		t.Fatalf("parsed observation differs: %+v", parsed)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parents := []artifact.ID{fixture.Model, fixture.Recipe, fixture.Environment}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "serving/fixture/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	batch, err := observation.Batch("serving/fixture/observation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	result, err := store.Query(t.Context(), overgodb.Query{
		Artifact: &fixture.Model, Follow: overgodb.FollowChildren, MaxDepth: 1,
		MediaType: ServingObservationMediaType, Schema: ServingObservationSchema, MaxResults: 8,
		Projection: overgodb.ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Artifacts) != 1 || result.Artifacts[0].ID != observation.ID || result.Truncated {
		t.Fatalf("indexed serving query = %+v", result)
	}
}

func TestServingObservationRefusesInvalidFacts(t *testing.T) {
	fixture := servingObservationFixture(t)
	refusals := []struct {
		name   string
		mutate func(*ServingObservation)
	}{
		{"model", func(value *ServingObservation) { value.Model = value.Recipe }},
		{"recipe", func(value *ServingObservation) { value.Recipe = value.Model }},
		{"environment", func(value *ServingObservation) { value.Environment = value.Model }},
		{"operation", func(value *ServingObservation) { value.Operation = value.Environment }},
		{"task", func(value *ServingObservation) { value.Task = recipe.Task("unknown") }},
		{"start", func(value *ServingObservation) { value.StartedUnixNS = 0 }},
		{"duration", func(value *ServingObservation) { value.MeasuredNS = math.MaxUint64 }},
		{"outcome", func(value *ServingObservation) { value.Outcome = Outcome("unknown") }},
		{"success failure", func(value *ServingObservation) { value.Failure = "failed" }},
		{"duplicate phase", func(value *ServingObservation) { value.Phases = append(value.Phases, value.Phases[0]) }},
		{"hardware order", func(value *ServingObservation) { value.Hardware = append(value.Hardware, value.Hardware[0]) }},
	}
	for _, refusal := range refusals {
		candidate := fixture
		candidate.Version = artifact.InitialDocumentVersion
		candidate.Phases = slices.Clone(fixture.Phases)
		candidate.Hardware = slices.Clone(fixture.Hardware)
		refusal.mutate(&candidate)
		if _, err := servingObservationCodec.New(candidate); err == nil {
			t.Errorf("%s: accepted", refusal.name)
		}
	}
	failed := fixture
	failed.Version = artifact.InitialDocumentVersion
	failed.Outcome, failed.Failure = OutcomeFailed, "execution_failed"
	if _, err := servingObservationCodec.New(failed); err != nil {
		t.Fatalf("failed observation refused: %v", err)
	}
}

func TestServingAttemptChainClassifiesRetryReselectionAndSpillover(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first := servingObservationFixture(t)
	first.Outcome, first.Failure = OutcomeFailed, "execution_failed"
	alternateRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "serving-alternate-recipe")
	compatibility := testutil.ArtifactID(t, artifact.KindEvidence, "serving-peer-compatibility")
	parents := []artifact.ID{first.Model, first.Recipe, alternateRecipe, first.Environment, compatibility}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "serving/attempt/authorities", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	first, err = PublishServingObservation(ctx, store, first)
	if err != nil || first.AttemptKind(nil) != ServingAttemptPrimary {
		t.Fatalf("primary attempt = (%+v, %v)", first, err)
	}
	next := func(recipeID artifact.ID, compatibilityID artifact.ID) ServingObservation {
		observation := servingObservationFixture(t)
		observation.Recipe, observation.Compatibility = recipeID, compatibilityID
		observation.Previous, observation.Attempt = first.ID, first.Attempt
		observation.Attempt++
		observation.StartedUnixNS = first.StartedUnixNS + int64(first.MeasuredNS)
		return observation
	}
	retry := next(first.Recipe, artifact.ID{})
	retry, err = PublishServingObservation(ctx, store, retry)
	if err != nil || retry.AttemptKind(&first) != ServingAttemptRetry {
		t.Fatalf("retry attempt = (%+v, %v)", retry, err)
	}
	reselection := next(alternateRecipe, artifact.ID{})
	if reselection.AttemptKind(&first) != ServingAttemptReselection {
		t.Fatalf("reselection kind = %q", reselection.AttemptKind(&first))
	}
	spillover := next(alternateRecipe, compatibility)
	if spillover.AttemptKind(&first) != ServingAttemptSpillover {
		t.Fatalf("spillover kind = %q", spillover.AttemptKind(&first))
	}
	if _, err := PublishServingObservation(ctx, store, spillover); err == nil {
		t.Fatal("branched serving attempt accepted")
	}
}

func TestOperationBoundServingObservationRequiresCurrentAlias(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	fixture := servingObservationFixture(t)
	parents := []artifact.ID{fixture.Model, fixture.Recipe, fixture.Environment}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, id := range parents {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "serving/operation-bound/authorities", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}

	current, err := PublishServingObservation(ctx, store, fixture)
	if err != nil {
		t.Fatal(err)
	}
	if required, err := RequireServingAttemptObservation(ctx, store, current.ID); err != nil || required.ID != current.ID {
		t.Fatalf("current operation-bound observation = (%s, %v)", required.ID, err)
	}

	foreign := fixture
	foreign.MeasuredNS++
	foreign, err = NewServingObservation(foreign)
	if err != nil {
		t.Fatal(err)
	}
	foreignContent, err := foreign.Content()
	if err != nil {
		t.Fatal(err)
	}
	foreignBatch, err := artifact.NewDocumentBatch(
		"serving/operation-bound/foreign", []artifact.Content{foreignContent}, foreign.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, foreignBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireServingObservation(ctx, store, foreign.ID); err != nil {
		t.Fatalf("well-formed historical observation was unreadable: %v", err)
	}
	if _, err := RequireServingAttemptObservation(ctx, store, foreign.ID); err == nil {
		t.Fatal("foreign observation sharing an operation ordinal was admitted")
	}

	unbound := fixture
	unbound.Operation = artifact.ID{}
	unbound.MeasuredNS += 2
	unbound, err = NewServingObservation(unbound)
	if err != nil {
		t.Fatal(err)
	}
	unboundBatch, err := unbound.Batch("serving/operation-bound/unbound")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, unboundBatch); err != nil {
		t.Fatal(err)
	}
	if _, err := RequireServingAttemptObservation(ctx, store, unbound.ID); err == nil {
		t.Fatal("serving observation without an operation was admitted")
	}
}
