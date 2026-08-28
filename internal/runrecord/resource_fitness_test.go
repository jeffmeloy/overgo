package runrecord

import (
	"encoding/json"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestResourceFitnessContract(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, name)
	}
	work := InteractionWork{
		SemanticTransitions: 1, Commits: 2, ArtifactReads: 3, BlobReads: 4,
		ScannedFacts: 5, ReturnedFacts: 6, Bytes: 7, Wakeups: 8, Retries: 9,
		ModelTurns: 10, ContextBytes: 11, RepeatedContextIDs: 12, ToolCalls: 13, Waits: 14,
	}
	fixture := ResourceFitness{
		Scope: ResourceScope{
			Surface:  SurfaceTraining,
			Model:    id(artifact.KindModel, "resource-model"),
			Hardware: id(artifact.KindEvidence, "resource-hardware"),
			Provider: id(artifact.KindProfile, "resource-provider"),
			Workload: id(artifact.KindRecipe, "resource-workload"),
			Attempt:  id(artifact.KindRun, "resource-attempt"),
		},
		Measures: []ResourceMeasure{
			{Metric: ResourceWallNS, Value: 29},
			{Metric: ResourceCostUnits, Value: 0},
			{Metric: ResourceInputTokens, Value: 1},
			{Metric: ResourceOutputTokens, Value: 2},
			{Metric: ResourceCacheReadTokens, Value: 3},
			{Metric: ResourceCacheWriteTokens, Value: 4},
			{Metric: ResourceInputBytes, Value: 5},
			{Metric: ResourceOutputBytes, Value: 6},
			{Metric: ResourceCacheLookups, Value: 7},
			{Metric: ResourceCacheHits, Value: 6},
			{Metric: ResourceCacheWrites, Value: 8},
			{Metric: ResourceActiveComputeNS, Value: 23},
			{Metric: ResourceCPUNS, Value: 31},
			{Metric: ResourceGPUNS, Value: 37},
			{Metric: ResourcePeakHostBytes, Value: 41},
			{Metric: ResourcePeakDeviceBytes, Value: 43},
			{Metric: ResourceDiskReadBytes, Value: 47},
			{Metric: ResourceDiskWriteBytes, Value: 53},
			{Metric: ResourceNetworkReceiveBytes, Value: 59},
			{Metric: ResourceNetworkSendBytes, Value: 61},
			{Metric: ResourceHostToDeviceBytes, Value: 67},
			{Metric: ResourceDeviceToHostBytes, Value: 71},
		},
		Interactions: &work,
	}
	fitness, err := NewResourceFitness(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if err := fitness.Validate(); err != nil {
		t.Fatal(err)
	}
	if observed, known := fitness.Measure(ResourceCostUnits); !known || observed != 0 {
		t.Fatalf("observed zero cost = (%d, %v)", observed, known)
	}
	if observed, known := fitness.Measure(ResourceMetric("unknown")); known || observed != 0 {
		t.Fatalf("unknown metric = (%d, %v)", observed, known)
	}
	if fitness.Interactions == nil || *fitness.Interactions != work {
		t.Fatalf("interaction owner not preserved: %+v", fitness.Interactions)
	}
	work.Retries++
	if fitness.Interactions.Retries != 9 {
		t.Fatal("resource fitness aliases caller interaction work")
	}
	encoded, err := json.Marshal(fitness)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ResourceFitness
	if err := json.Unmarshal(encoded, &decoded); err != nil || decoded.Validate() != nil || !reflect.DeepEqual(decoded, fitness) {
		t.Fatalf("canonical round trip differs: decoded=%+v err=%v", decoded, err)
	}

	withoutZero := fitness
	withoutZero.Measures = slices.DeleteFunc(slices.Clone(fitness.Measures), func(measure ResourceMeasure) bool {
		return measure.Metric == ResourceCostUnits
	})
	unknownID, err := artifact.JSONID(artifact.KindEvidence, withoutZero)
	if err != nil {
		t.Fatal(err)
	}
	zeroID, err := artifact.JSONID(artifact.KindEvidence, fitness)
	if err != nil || zeroID == unknownID {
		t.Fatalf("known zero collapsed with unknown: zero=%s unknown=%s err=%v", zeroID, unknownID, err)
	}
	unknownInteractions := cloneResourceFitness(fitness)
	unknownInteractions.Interactions = nil
	unknownInteractionID, err := artifact.JSONID(artifact.KindEvidence, unknownInteractions)
	if err != nil {
		t.Fatal(err)
	}
	observedZeroInteractions := cloneResourceFitness(unknownInteractions)
	observedZeroInteractions.Interactions = &InteractionWork{}
	observedInteractionID, err := artifact.JSONID(artifact.KindEvidence, observedZeroInteractions)
	if err != nil || observedInteractionID == unknownInteractionID {
		t.Fatalf("known-zero interactions collapsed with unknown: observed=%s unknown=%s err=%v", observedInteractionID, unknownInteractionID, err)
	}

	baseID := zeroID
	mutations := []struct {
		name string
		set  func(*ResourceFitness)
	}{
		{"model", func(value *ResourceFitness) { value.Scope.Model = id(artifact.KindModel, "other-model") }},
		{"hardware", func(value *ResourceFitness) { value.Scope.Hardware = id(artifact.KindEvidence, "other-hardware") }},
		{"provider", func(value *ResourceFitness) { value.Scope.Provider = id(artifact.KindProfile, "other-provider") }},
		{"workload", func(value *ResourceFitness) { value.Scope.Workload = id(artifact.KindDataset, "other-workload") }},
		{"attempt", func(value *ResourceFitness) { value.Scope.Attempt = id(artifact.KindEvidence, "other-attempt") }},
	}
	for _, mutation := range mutations {
		t.Run("axis "+mutation.name, func(t *testing.T) {
			candidate := cloneResourceFitness(fitness)
			mutation.set(&candidate)
			candidate, err = NewResourceFitness(candidate)
			if err != nil {
				t.Fatal(err)
			}
			candidateID, err := artifact.JSONID(artifact.KindEvidence, candidate)
			if err != nil || candidateID == baseID {
				t.Fatalf("axis collapsed: id=%s err=%v", candidateID, err)
			}
		})
	}

	wantAuthorities := []artifact.ID{
		fitness.Scope.Model, fitness.Scope.Hardware, fitness.Scope.Provider,
		fitness.Scope.Workload, fitness.Scope.Attempt,
	}
	if !slices.Equal(fitness.Authorities(), wantAuthorities) {
		t.Fatalf("authorities = %+v", fitness.Authorities())
	}
	if reflect.TypeOf(ResourceMeasure{}.Value).Kind() != reflect.Uint64 {
		t.Fatal("resource value is not integer uint64")
	}
	var fractional ResourceFitness
	if err := json.Unmarshal([]byte(`{"version":1,"scope":{},"measures":[{"metric":"cost_units","value":0.5}]}`), &fractional); err == nil {
		t.Fatal("fractional cost accepted")
	}

	servingFixture := servingObservationFixture(t)
	servingFixture.Resources.PeakHostBytes = 0
	serving, err := NewServingObservation(servingFixture)
	if err != nil {
		t.Fatal(err)
	}
	servingFitness, err := serving.ResourceFitness(
		artifact.ID{}, nil,
		ResourceWallNS, ResourceInputTokens, ResourceOutputTokens, ResourceInputBytes, ResourceOutputBytes,
		ResourcePeakHostBytes, ResourcePeakDeviceBytes, ResourceHostToDeviceBytes, ResourceDeviceToHostBytes,
	)
	if err != nil {
		t.Fatal(err)
	}
	if servingFitness.Scope.Surface != SurfaceServing || servingFitness.Scope.Model != serving.Model ||
		servingFitness.Scope.Hardware != serving.Environment || servingFitness.Scope.Workload != serving.Recipe ||
		servingFitness.Scope.Attempt != serving.ID {
		t.Fatalf("serving scope = %+v", servingFitness.Scope)
	}
	if peak, known := servingFitness.Measure(ResourcePeakHostBytes); !known || peak != 0 {
		t.Fatalf("observed-zero serving peak = (%d, %v)", peak, known)
	}
	if _, err := serving.ResourceFitness(artifact.ID{}, nil, ResourceNetworkSendBytes); err == nil {
		t.Fatal("serving adapter accepted an unowned metric")
	}

	refusals := []struct {
		name   string
		mutate func(*ResourceFitness)
	}{
		{"surface", func(value *ResourceFitness) { value.Scope.Surface = "foreign" }},
		{"model kind", func(value *ResourceFitness) { value.Scope.Model = id(artifact.KindFile, "wrong-model") }},
		{"hardware kind", func(value *ResourceFitness) { value.Scope.Hardware = id(artifact.KindModel, "wrong-hardware") }},
		{"provider kind", func(value *ResourceFitness) { value.Scope.Provider = id(artifact.KindEvidence, "wrong-provider") }},
		{"workload kind", func(value *ResourceFitness) { value.Scope.Workload = id(artifact.KindOutput, "wrong-workload") }},
		{"attempt kind", func(value *ResourceFitness) { value.Scope.Attempt = id(artifact.KindFile, "wrong-attempt") }},
		{"collapsed axes", func(value *ResourceFitness) { value.Scope.Attempt = value.Scope.Hardware }},
		{"foreign metric", func(value *ResourceFitness) {
			value.Measures = append(value.Measures, ResourceMeasure{Metric: "joules", Value: 1})
		}},
		{"duplicate metric", func(value *ResourceFitness) { value.Measures = append(value.Measures, value.Measures[0]) }},
		{"hits over lookups", func(value *ResourceFitness) {
			for index := range value.Measures {
				if value.Measures[index].Metric == ResourceCacheHits {
					value.Measures[index].Value = 8
				}
			}
		}},
		{"empty", func(value *ResourceFitness) { value.Measures, value.Interactions = nil, nil }},
	}
	for _, refusal := range refusals {
		t.Run("refuses "+refusal.name, func(t *testing.T) {
			candidate := cloneResourceFitness(fitness)
			refusal.mutate(&candidate)
			if _, err := NewResourceFitness(candidate); err == nil {
				t.Fatal("accepted")
			}
		})
	}
}
