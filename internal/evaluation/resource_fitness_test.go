package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

type terminalResourceRuntime struct{ failure error }

func (runtime terminalResourceRuntime) Generate(
	context.Context,
	string,
	inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	return nil, "", runtime.failure
}

func (runtime terminalResourceRuntime) ScoreContinuations(
	context.Context,
	string,
	[]string,
) ([]sequencescore.Score, error) {
	return nil, runtime.failure
}

func TestResourceFitnessContract(t *testing.T) {
	model := planID(t, artifact.KindModel, "resource model")
	recipe := planID(t, artifact.KindRecipe, "resource recipe")
	workload := planID(t, artifact.KindProfile, "compiled evaluation plan")
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture",
		Backend: "fixture", Driver: "fixture",
	})
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{workload}, []artifact.ID{planID(t, artifact.KindOutput, "resource output")}, "",
		runtimeLedgerCommit, environment.ID, 31, []runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 31}},
	)
	if err != nil {
		t.Fatal(err)
	}
	campaign := &Campaign{
		identity:    modelrecipe.ProgramIdentity{Model: model, Recipe: recipe},
		environment: environment,
	}
	fitness, err := campaign.resourceFitness(workload, run)
	if err != nil {
		t.Fatal(err)
	}
	if fitness.Scope.Surface != runrecord.SurfaceEvaluation || fitness.Scope.Model != model ||
		fitness.Scope.Hardware != environment.ID || fitness.Scope.Provider.Valid() ||
		fitness.Scope.Workload != workload || fitness.Scope.Attempt != run.ID || run.Recipe != recipe {
		t.Fatalf("fitness scope = %+v; run recipe = %s", fitness.Scope, run.Recipe)
	}
	if wall, observed := fitness.Measure(runrecord.ResourceWallNS); !observed || wall != run.MeasuredNS {
		t.Fatalf("wall = %d, observed %t", wall, observed)
	}
	if tokens, observed := fitness.Measure(runrecord.ResourceInputTokens); observed || tokens != 0 {
		t.Fatalf("unknown input tokens = %d, observed %t", tokens, observed)
	}

	observedZero := fitness
	observedZero.Measures = append(observedZero.Measures, runrecord.ResourceMeasure{
		Metric: runrecord.ResourceInputTokens, Value: 0,
	})
	observedZero, err = runrecord.NewResourceFitness(observedZero)
	if err != nil {
		t.Fatal(err)
	}
	if tokens, observed := observedZero.Measure(runrecord.ResourceInputTokens); !observed || tokens != 0 {
		t.Fatalf("observed-zero input tokens = %d, observed %t", tokens, observed)
	}

	foreign := run
	foreign.Recipe = planID(t, artifact.KindRecipe, "foreign resource recipe")
	foreign, err = runrecord.NewBoundRun(
		foreign.Recipe, foreign.Outcome, foreign.Inputs, foreign.Outputs, foreign.Failure,
		foreign.CodeCommit, foreign.Environment, foreign.MeasuredNS, foreign.Phases,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := campaign.resourceFitness(workload, foreign); err == nil {
		t.Fatal("accepted a run from another recipe")
	}
	if _, err := campaign.resourceFitness(planID(t, artifact.KindProfile, "foreign workload"), run); err == nil {
		t.Fatal("accepted a workload outside the run inputs")
	}

	for _, terminal := range []struct {
		name    string
		failure error
		outcome runrecord.Outcome
	}{
		{name: "failure", failure: errors.New("fixture failure"), outcome: runrecord.OutcomeFailed},
		{name: "cancellation", failure: context.Canceled, outcome: runrecord.OutcomeCancelled},
	} {
		t.Run("terminal "+terminal.name, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			definition := planID(t, artifact.KindModelDefinition, "resource definition "+terminal.name)
			if _, err := store.Commit(context.Background(), artifact.Batch{
				Key: "fixture/resource-" + terminal.name,
				Artifacts: []artifact.Descriptor{
					{ID: model}, {ID: definition}, {ID: recipe},
				},
			}); err != nil {
				t.Fatal(err)
			}
			terminalCampaign, err := NewCampaign(
				store, terminalResourceRuntime{failure: terminal.failure},
				modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: recipe},
				environment, runtimeLedgerCommit,
			)
			if err != nil {
				t.Fatal(err)
			}
			source, err := json.Marshal(exactFixture())
			if err != nil {
				t.Fatal(err)
			}
			suite, err := CompileSuite(source, terminalCampaign.Authorities())
			if err != nil {
				t.Fatal(err)
			}
			result, evaluateErr := terminalCampaign.Evaluate(context.Background(), suite)
			if evaluateErr == nil {
				t.Fatal("terminal evaluation succeeded")
			}
			storedRun, err := runrecord.RequireRun(context.Background(), store, result.Run)
			if err != nil || storedRun.Outcome != terminal.outcome {
				t.Fatalf("terminal run = %+v, err %v", storedRun, err)
			}
			wall, observed := result.Resources.Measure(runrecord.ResourceWallNS)
			if !observed || wall == 0 || result.Resources.Scope.Workload != suite.Plan().Identity() ||
				result.Resources.Scope.Attempt != result.Run {
				t.Fatalf("terminal resources = %+v", result.Resources)
			}
		})
	}
}
