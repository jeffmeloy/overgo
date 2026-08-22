package runrecord

import (
	"context"
	"math"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

const fixtureCodeCommit = "0123456789abcdef0123456789abcdef01234567"

func TestRunAndEvaluationRoundTrip(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
	inputID := testutil.ArtifactID(t, artifact.KindFile, "input")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "output")
	run, err := NewRun(recipeID, OutcomeSucceeded, []artifact.ID{inputID}, []artifact.ID{outputID}, "")
	if err != nil {
		t.Fatal(err)
	}
	content, err := run.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsedRun, err := ParseRun(content.Data)
	if err != nil || parsedRun.ID != run.ID {
		t.Fatalf("run round trip = (%+v, %v)", parsedRun, err)
	}
	evaluation, err := NewEvaluation(recipeID, run.ID, datasetID, []Metric{
		{Name: "latency", Value: 12.5, Unit: "ms", Direction: DirectionMinimize},
		{Name: "quality", Value: 0.9, Direction: DirectionMaximize},
	})
	if err != nil {
		t.Fatal(err)
	}
	evaluationContent, err := evaluation.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsedEvaluation, err := ParseEvaluation(evaluationContent.Data)
	if err != nil || parsedEvaluation.ID != evaluation.ID || len(evaluation.Lineage()) != 3 {
		t.Fatalf("evaluation round trip = (%+v, %v)", parsedEvaluation, err)
	}
	if len(run.Lineage()) != 3 {
		t.Fatalf("run lineage = %+v", run.Lineage())
	}
}

func TestRunAndEvaluationPersistWithLineage(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
	inputID := testutil.ArtifactID(t, artifact.KindFile, "input")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "output")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/facts", Artifacts: []artifact.Descriptor{
		{ID: recipeID}, {ID: datasetID}, {ID: inputID}, {ID: outputID},
	}}); err != nil {
		t.Fatal(err)
	}
	run, err := NewRun(recipeID, OutcomeSucceeded, []artifact.ID{inputID}, []artifact.ID{outputID}, "")
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("fixture/run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	storedRun, err := RequireRun(ctx, store, run.ID)
	if err != nil || storedRun.ID != run.ID {
		t.Fatalf("stored run = (%+v, %v)", storedRun, err)
	}
	evaluation, err := NewEvaluation(recipeID, run.ID, datasetID, []Metric{{
		Name: "quality", Value: 1, Direction: DirectionMaximize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = evaluation.Batch("fixture/evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	storedEvaluation, err := RequireEvaluation(ctx, store, evaluation.ID)
	if err != nil || storedEvaluation.ID != evaluation.ID {
		t.Fatalf("stored evaluation = (%+v, %v)", storedEvaluation, err)
	}
	parents, err := store.Parents(ctx, evaluation.ID)
	if err != nil || len(parents) != 3 {
		t.Fatalf("evaluation parents = (%+v, %v)", parents, err)
	}
}

func TestRunAndEvaluationRejectInvalidFacts(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "dataset")
	runID := testutil.ArtifactID(t, artifact.KindRun, "run")
	if _, err := NewRun(recipeID, OutcomeSucceeded, nil, nil, ""); err == nil {
		t.Fatal("successful run without output accepted")
	}
	if _, err := NewRun(recipeID, OutcomeFailed, nil, nil, ""); err == nil {
		t.Fatal("failed run without failure code accepted")
	}
	if _, err := NewEvaluation(recipeID, runID, datasetID, []Metric{{
		Name: "quality", Value: math.NaN(), Direction: DirectionMaximize,
	}}); err == nil {
		t.Fatal("non-finite metric accepted")
	}
}

func TestBoundRunEnvironmentAndPhasesRoundTrip(t *testing.T) {
	environment, err := NewEnvironment(Environment{
		Host: "fixture-host", OS: "windows", Arch: "amd64",
		Device: "RTX 4090", Backend: "cuda", Driver: "591.44", Runtime: "go1.25",
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsedEnvironment, err := environmentCodec.Parse(content.Data)
	if err != nil || parsedEnvironment != environment {
		t.Fatalf("environment round trip = (%+v, %v)", parsedEnvironment, err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bound-recipe")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "bound-output")
	run, err := NewBoundRun(
		recipeID, OutcomeSucceeded, nil, []artifact.ID{outputID}, "",
		fixtureCodeCommit, environment.ID, 100, []PhaseMetric{
			{Phase: PhaseDecode, DurationNS: 55},
			{Phase: PhasePrefill, DurationNS: 35},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if run.MeasuredNS != 100 || !slices.Equal(run.Phases, []PhaseMetric{
		{Phase: PhaseDecode, DurationNS: 55},
		{Phase: PhasePrefill, DurationNS: 35},
	}) {
		t.Fatalf("bound timing = (%d, %+v)", run.MeasuredNS, run.Phases)
	}
	content, err = run.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsedRun, err := ParseRun(content.Data)
	if err != nil || parsedRun.ID != run.ID || parsedRun.MeasuredNS != run.MeasuredNS ||
		!slices.Equal(parsedRun.Phases, run.Phases) {
		t.Fatalf("bound run round trip = (%+v, %v)", parsedRun, err)
	}
	if len(run.Lineage()) != 3 {
		t.Fatalf("bound run lineage = %+v", run.Lineage())
	}
}

func TestBoundRunRejectsUncontrolledTimingFacts(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bound-recipe")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "bound-output")
	environmentID := testutil.ArtifactID(t, artifact.KindEvidence, "environment")
	newRun := func(commit string, phases []PhaseMetric) error {
		_, err := NewBoundRun(
			recipeID, OutcomeSucceeded, nil, []artifact.ID{outputID}, "", commit,
			environmentID, 10, phases,
		)
		return err
	}
	if err := newRun("working-tree", []PhaseMetric{{Phase: PhaseDecode, DurationNS: 1}}); err == nil {
		t.Fatal("non-commit run accepted")
	}
	if err := newRun(fixtureCodeCommit, []PhaseMetric{{Phase: "other", DurationNS: 1}}); err == nil {
		t.Fatal("unknown phase accepted")
	}
	if err := newRun(fixtureCodeCommit, []PhaseMetric{
		{Phase: PhaseDecode, DurationNS: 1}, {Phase: PhaseDecode, DurationNS: 2},
	}); err == nil {
		t.Fatal("duplicate phase accepted")
	}
	run, err := NewBoundRun(
		recipeID, OutcomeSucceeded, nil, []artifact.ID{outputID}, "", fixtureCodeCommit,
		environmentID, 10, []PhaseMetric{{Phase: PhaseDecode, DurationNS: 12}},
	)
	if err != nil || run.MeasuredNS != 10 || !slices.Equal(run.Phases, []PhaseMetric{{
		Phase: PhaseDecode, DurationNS: 12,
	}}) {
		t.Fatalf("over-attributed timing = (%+v, %v)", run, err)
	}
}

func TestBoundRunPersistsEnvironmentLineage(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	environment, err := environmentCodec.New(Environment{
		Version: artifact.InitialDocumentVersion, Host: "fixture-host", OS: "linux", Arch: "amd64",
		Device: "A100", Backend: "cuda", Driver: "580.65", Runtime: "go1.25",
	})
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "bound-recipe")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "bound-output")
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/bound-facts",
		Artifacts: []artifact.Descriptor{
			{ID: recipeID}, {ID: outputID}, environmentContent.Descriptor,
		},
		Contents: []artifact.Content{environmentContent},
	}); err != nil {
		t.Fatal(err)
	}
	run, err := NewBoundRun(
		recipeID, OutcomeSucceeded, nil, []artifact.ID{outputID}, "", fixtureCodeCommit,
		environment.ID, 100, []PhaseMetric{{Phase: PhaseDecode, DurationNS: 90}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := run.Batch("fixture/bound-run")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(ctx, run.ID)
	if err != nil || len(parents) != 2 {
		t.Fatalf("bound run parents = (%+v, %v)", parents, err)
	}
}
