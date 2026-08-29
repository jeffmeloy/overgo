package inference

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type compositeGenerationObserverFixture struct{}

func (compositeGenerationObserverFixture) Observe(
	ctx context.Context,
	_ composition.CompositeGenerationArm,
	execute func(context.Context) (reference.Value, error),
) (reference.Value, CompositeGenerationMeasurement, error) {
	output, err := execute(ctx)
	return output, CompositeGenerationMeasurement{
		StartedUnixNS: 1_700_000_000_000_000_000,
		MeasuredNS:    100,
		Usage:         runrecord.ServingUsage{InputTokens: 4, OutputTokens: 2},
		Resources:     runrecord.ServingResources{PeakHostBytes: 1024, PeakDeviceBytes: 2048},
		Phases:        []runrecord.PhaseMetric{{Phase: runrecord.PhaseDecode, DurationNS: 80}},
		Hardware: []runrecord.ServingHardwareSample{
			{Stage: runrecord.ServingHardwareStart, DeviceCurrentBytes: 1024, DevicePeakBytes: 2048},
			{Stage: runrecord.ServingHardwareFinish, ElapsedNS: 100, DeviceCurrentBytes: 1024, DevicePeakBytes: 2048},
		},
	}, err
}

type compositeGenerationMetricFixture struct{}

func (compositeGenerationMetricFixture) Evaluate(
	_ context.Context,
	arm composition.CompositeGenerationArm,
	_ artifact.ID,
	output reference.Value,
) ([]runrecord.Metric, error) {
	return []runrecord.Metric{{
		Name: "quality-score", Value: float64(len(output.Data)) + float64(len(arm)),
		Direction: runrecord.DirectionMaximize,
	}}, nil
}

func TestCompositeGenerationActiveRecipe(t *testing.T) {
	runtime, store, _, request, _ := compositeGenerationExecutionFixture(t)
	request.TargetBaselineRecipe = runtime.plan.CompositionRecipe
	if _, err := runtime.GenerateEvidence(t.Context(), store, request); err == nil {
		t.Fatal("control recipe aliasing the active composition was admitted")
	}
	if runtime.plan.CompositionRecipe.Kind() != artifact.KindRecipe || runtime.plan.ValidateIdentity() != nil {
		t.Fatalf("active plan = %+v", runtime.plan)
	}
}

func TestCompositeGenerationExecution(t *testing.T) {
	runtime, store, _, request, source := compositeGenerationExecutionFixture(t)
	result, err := runtime.GenerateEvidence(t.Context(), store, request)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Commit.Valid() || len(result.Evidence.Trials) != len(compositeGenerationArms) ||
		result.Evidence.CompositionRecipe != runtime.plan.CompositionRecipe ||
		result.Evidence.ExecutionPlan != runtime.plan.ID || source.calls != 3 {
		t.Fatalf("execution result=%+v source calls=%d", result, source.calls)
	}
	loaded, err := (composition.CompositeGenerationEvidenceAuthority{}).Load(
		t.Context(), store, result.Evidence.ID,
	)
	if err != nil || loaded.ID != result.Evidence.ID {
		t.Fatalf("stored evidence = %+v, %v", loaded, err)
	}
}

func TestCompositeGenerationOutputLineage(t *testing.T) {
	runtime, store, _, request, _ := compositeGenerationExecutionFixture(t)
	result, err := runtime.GenerateEvidence(t.Context(), store, request)
	if err != nil {
		t.Fatal(err)
	}
	trial := result.Evidence.Trials[0]
	parents, err := store.Parents(t.Context(), trial.Output)
	if err != nil {
		t.Fatal(err)
	}
	want := map[artifact.ID]artifact.Relation{
		trial.Input: artifact.RelationDerivedFrom,
		trial.Run:   artifact.RelationProducedBy,
	}
	for _, edge := range parents {
		delete(want, edge.Parent)
	}
	if len(want) != 0 {
		t.Fatalf("output lineage lacks %+v: %+v", want, parents)
	}
	children, err := store.Children(t.Context(), result.Evidence.ID)
	if err != nil || len(children) != 0 {
		t.Fatalf("unexpected evidence descendants = %+v, %v", children, err)
	}
}

func TestCompositeGenerationResourceEvidence(t *testing.T) {
	runtime, store, _, request, _ := compositeGenerationExecutionFixture(t)
	result, err := runtime.GenerateEvidence(t.Context(), store, request)
	if err != nil {
		t.Fatal(err)
	}
	for _, trial := range result.Evidence.Trials {
		run, runErr := runrecord.RequireRun(t.Context(), store, trial.Run)
		evaluation, evaluationErr := runrecord.RequireEvaluation(t.Context(), store, trial.Evaluation)
		observation, observationErr := runrecord.RequireServingObservation(t.Context(), store, trial.Observation)
		if runErr != nil || evaluationErr != nil || observationErr != nil ||
			run.MeasuredNS != observation.MeasuredNS || evaluation.Run != run.ID ||
			observation.Resources.PeakDeviceBytes == 0 || len(evaluation.Metrics) != 1 {
			t.Fatalf("resource graph run=%+v evaluation=%+v observation=%+v errors=%v/%v/%v",
				run, evaluation, observation, runErr, evaluationErr, observationErr)
		}
	}
}

func compositeGenerationExecutionFixture(
	t *testing.T,
) (*ProductionComposition, *overgodb.Store, composition.CompositionAuthority, CompositeGenerationExecutionRequest, *bridgeSourceFixture) {
	t.Helper()
	store, authority := productionCompositionFixture(t, true)
	source := &bridgeSourceFixture{
		model: authority.Recipe.SourceModel,
		value: inferenceBridgeValue(t, tensor.MustShape(2, 2), []float32{1, 2, 3, 4}),
	}
	target := &bridgeTargetFixture{
		model:     authority.Recipe.TargetModel,
		embedding: inferenceBridgeValue(t, tensor.MustShape(3, 2), []float32{5, 6, 7, 8, 9, 10}),
	}
	first := inferenceBridgeValue(t, tensor.MustShape(2, 3), []float32{1, 0, 0, 1, 1, 1})
	bias := inferenceBridgeValue(t, tensor.MustShape(3), []float32{1, 2, 3})
	runtime, err := OpenProductionComposition(
		t.Context(), store, authority.Recipe.SourceModel, authority.Recipe.TargetModel,
		authority.Recipe.Task, &bridgeRuntimeResourcesFixture{
			source: source, target: target,
			weights: RepresentationBridgeWeights{First: &first, FirstBias: &bias},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	request := CompositeGenerationExecutionRequest{
		Key:                  "fixture/composite-generation/execution",
		TargetBaselineRecipe: id(artifact.KindRecipe, "target-only generation"),
		SourceAblatedRecipe:  id(artifact.KindRecipe, "source-ablated generation"),
		BridgeAblatedRecipe:  id(artifact.KindRecipe, "bridge-ablated generation"),
		Dataset:              id(artifact.KindDataset, "composite generation dataset"),
		HeldOutSplit:         authority.Promotion.HeldOutSplit,
		Evaluator:            id(artifact.KindEvidence, "composite generation evaluator"),
		Environment:          id(artifact.KindEvidence, "composite generation environment"),
		CodeCommit:           strings.Repeat("a", 40),
		Cases: []CompositeGenerationCase{{
			Seed: 17, Input: id(artifact.KindOutput, "composite generation input"),
			SourceTokens: []tokenizer.TokenID{1, 2}, TargetTokens: []tokenizer.TokenID{3, 4},
		}},
		Observer: compositeGenerationObserverFixture{}, MetricEvaluator: compositeGenerationMetricFixture{},
	}
	authorities := []artifact.ID{
		request.TargetBaselineRecipe, request.SourceAblatedRecipe, request.BridgeAblatedRecipe,
		request.Dataset, request.Evaluator, request.Environment, request.Cases[0].Input,
	}
	descriptors := make([]artifact.Descriptor, len(authorities))
	for index, authorityID := range authorities {
		descriptors[index] = artifact.Descriptor{ID: authorityID, Size: 1}
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/composite-generation/authorities", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}
	return runtime, store, authority, request, source
}
