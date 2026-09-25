package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestComparisonRefusesMixedRealizedResidency(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		t.Helper()
		return testutil.ArtifactID(t, kind, "realized residency "+name)
	}
	model := id(artifact.KindModel, "model")
	workload := id(artifact.KindRecipe, "workload")
	hardware := id(artifact.KindEvidence, "hardware")
	stream := func(name string, actual recipe.RealizedResidency, wall uint64) ObservationStream {
		t.Helper()
		scope := ResourceScope{
			Surface: SurfaceServing, Model: model, Hardware: hardware, Workload: workload,
			Attempt: id(artifact.KindRun, name+" attempt"), RealizedResidency: actual,
		}
		aggregate, err := NewResourceFitness(ResourceFitness{
			Scope: scope, Measures: []ResourceMeasure{{Metric: ResourceWallNS, Value: wall}},
		})
		if err != nil {
			t.Fatal(err)
		}
		return ObservationStream{
			Scope: scope, Aggregate: aggregate,
			SummaryIDs: []artifact.ID{id(artifact.KindEvidence, name+" summary")},
			ChunkIDs:   []artifact.ID{id(artifact.KindFile, name+" chunk")},
			Coverage: ObservationStreamCoverage{
				RawBytes: 1, Samples: 1, MeasuredSamples: 1,
				FirstOrdinal: 1, LastOrdinal: 1, FirstElapsedNS: 1, LastElapsedNS: 1,
				Metrics: []ObservationMetricSamples{{Metric: ResourceWallNS, Samples: 1}},
				Kinds:   []ObservationKindSamples{{Kind: ObservationSampleExecution, Samples: 1}},
			},
		}
	}
	lane := func(before, after recipe.RealizedResidency) ResourceFitnessLane {
		return ResourceFitnessLane{
			Name: "serving", RequiredMetrics: []ResourceMetric{ResourceWallNS},
			Baseline: stream("baseline", before, 10), Candidate: stream("candidate", after, 9),
		}
	}
	for _, tc := range []struct {
		name          string
		before, after recipe.RealizedResidency
	}{
		{"legacy baseline", "", recipe.RealizedDeviceNative},
		{"legacy candidate", recipe.RealizedDeviceNative, ""},
		{"mixed fallback", recipe.RealizedDeviceNative, recipe.RealizedOOMStreamed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewResourceFitnessComparison([]ResourceFitnessLane{lane(tc.before, tc.after)}); err == nil || !strings.Contains(err.Error(), "residency") {
				t.Fatalf("incomparable serving residency accepted: %v", err)
			}
		})
	}
	compared, err := NewResourceFitnessComparison([]ResourceFitnessLane{lane(recipe.RealizedDeviceNative, recipe.RealizedDeviceNative)})
	if err != nil || !compared.StrictImprovement {
		t.Fatalf("matched serving comparison = %+v, %v", compared, err)
	}
	serving := servingObservationFixture(t)
	serving.RealizedResidency = recipe.RealizedOOMStreamed
	serving, err = NewServingObservation(serving)
	if err != nil {
		t.Fatal(err)
	}
	fitness, err := serving.ResourceFitness(artifact.ID{}, nil, ResourceWallNS)
	if err != nil || fitness.Scope.RealizedResidency != recipe.RealizedOOMStreamed {
		t.Fatalf("serving outcome lost in resource observation: %+v, %v", fitness.Scope, err)
	}
}
