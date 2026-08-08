package runrecord

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMADRegressionAdvisoryReplaysKnownSeries(t *testing.T) {
	values := []float64{100, 101, 99, 100, 130}
	observations := make([]Observation, len(values))
	for index, value := range values {
		observations[index] = advisoryObservation(t, uint64(index+1), value, uint64(50+index), uint64(40+index))
	}
	advisory, triggered, err := DetectRegression(observations, "latency", 4, 6)
	if err != nil || !triggered {
		t.Fatalf("advisory = (%+v, %t, %v)", advisory, triggered, err)
	}
	if advisory.BaselineMedian != 100 || advisory.MAD != 0.5 || advisory.Surprise() != 60 {
		t.Fatalf("robust score = median %g MAD %g surprise %g", advisory.BaselineMedian, advisory.MAD, advisory.Surprise())
	}
	if len(advisory.PhaseDeltas) != 2 {
		t.Fatalf("phase deltas = %+v", advisory.PhaseDeltas)
	}
	content, err := advisory.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseAdvisory(content)
	if err != nil || parsed.ID != advisory.ID || parsed.Surprise() != advisory.Surprise() {
		t.Fatalf("advisory round trip = (%+v, %v)", parsed, err)
	}
}

func TestMADRegressionAdvisoryRespectsDirectionAndZeroMAD(t *testing.T) {
	values := []float64{10, 10, 10, 5}
	observations := make([]Observation, len(values))
	for index, value := range values {
		observation := advisoryObservation(t, uint64(index+1), value, 5, 5)
		observation.Evaluation.Metrics[0].Direction = DirectionMinimize
		observation.Evaluation, _ = NewEvaluation(
			observation.Run.Recipe, observation.Run.ID, observation.Evaluation.Dataset,
			observation.Evaluation.Metrics,
		)
		observations[index] = observation
	}
	if _, triggered, err := DetectRegression(observations, "latency", 3, 3); err != nil || triggered {
		t.Fatalf("improvement triggered = (%t, %v)", triggered, err)
	}
	observations[len(observations)-1] = advisoryObservation(t, 4, 11, 6, 5)
	advisory, triggered, err := DetectRegression(observations, "latency", 3, 3)
	if err != nil || !triggered || !math.IsInf(advisory.Surprise(), 1) {
		t.Fatalf("zero-MAD advisory = (%+v, %t, %v)", advisory, triggered, err)
	}
}

func TestMADRegressionAdvisoryRejectsMixedEnvironment(t *testing.T) {
	observations := []Observation{
		advisoryObservation(t, 1, 1, 1, 1), advisoryObservation(t, 2, 1, 1, 1),
		advisoryObservation(t, 3, 1, 1, 1), advisoryObservation(t, 4, 2, 1, 1),
	}
	otherEnvironment := testutil.ArtifactID(t, artifact.KindEvidence, "other-environment")
	prior := observations[1]
	otherRun, err := NewBoundRun(
		prior.Run.Recipe, prior.Run.Outcome, prior.Run.Inputs, prior.Run.Outputs, prior.Run.Failure,
		prior.Run.CodeCommit, otherEnvironment, prior.Run.MeasuredNS, prior.Run.Phases,
	)
	if err != nil {
		t.Fatal(err)
	}
	otherEvaluation, err := NewEvaluation(
		prior.Evaluation.Recipe, otherRun.ID, prior.Evaluation.Dataset, prior.Evaluation.Metrics,
	)
	if err != nil {
		t.Fatal(err)
	}
	observations[1].Run, observations[1].Evaluation = otherRun, otherEvaluation
	if _, _, err := DetectRegression(observations, "latency", 3, 3); err == nil {
		t.Fatal("mixed environment series accepted")
	}
}

func advisoryObservation(t *testing.T, sequence uint64, value float64, prefill, decode uint64) Observation {
	t.Helper()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "advisory-recipe")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "advisory-environment")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "output-"+string(rune('a'+sequence)))
	run, err := NewBoundRun(
		recipeID, OutcomeSucceeded, nil, []artifact.ID{outputID}, "", fixtureCodeCommit,
		environmentID, prefill+decode+10, []PhaseMetric{
			{Phase: PhasePrefill, DurationNS: prefill},
			{Phase: PhaseDecode, DurationNS: decode},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := NewEvaluation(
		recipeID, run.ID, testutil.ArtifactID(t, artifact.KindDataset, "advisory-dataset"),
		[]Metric{{Name: "latency", Value: value, Unit: "ms", Direction: DirectionMinimize}},
	)
	if err != nil {
		t.Fatal(err)
	}
	return Observation{Sequence: sequence, Run: run, Evaluation: evaluation}
}
