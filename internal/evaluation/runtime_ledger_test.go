package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

const runtimeLedgerCommit = "0123456789abcdef0123456789abcdef01234567"

type observedRuntime struct {
	generateCalls int
}

func (runtime *observedRuntime) Generate(
	ctx context.Context,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	runtime.generateCalls++
	return (exactGenerator{pieces: []string{"o", "k"}}).Generate(ctx, prompt, options)
}

func (*observedRuntime) ScoreContinuations(
	context.Context,
	string,
	[]string,
) ([]sequencescore.Score, error) {
	return nil, nil
}

func TestIsolatedModelPerformanceEvidence(t *testing.T) {
	testCampaignPublication(t, LifecycleIsolated, false)
}

func TestResidentCampaignPublication(t *testing.T) {
	t.Run("success", func(t *testing.T) { testCampaignPublication(t, LifecycleResident, false) })
	t.Run("publication recovery", func(t *testing.T) { testCampaignPublication(t, LifecycleResident, true) })
}

type publicationFaultStore struct {
	*overgodb.Store
	fail bool
}

var errPublicationFixture = errors.New("fixture publication failure")

func (store *publicationFaultStore) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if store.fail && strings.HasPrefix(batch.Key, "evaluation/run/") {
		return artifact.CommitID{}, errPublicationFixture
	}
	return store.Store.Commit(ctx, batch)
}

func testCampaignPublication(t *testing.T, lifecycle Lifecycle, failPublication bool) {
	t.Helper()
	ctx := t.Context()
	root := t.TempDir()
	database, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	store := &publicationFaultStore{Store: database, fail: failPublication}
	t.Cleanup(func() { _ = store.Close() })
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "model"),
		Definition: planID(t, artifact.KindModelDefinition, "definition"),
		Recipe:     planID(t, artifact.KindRecipe, "active recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/active-recipe",
		Artifacts: []artifact.Descriptor{
			{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe},
		},
	}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture",
		Backend: "fixture", Driver: "fixture",
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &observedRuntime{}
	campaign, err := newCampaign(store, runtime, identity, environment, runtimeLedgerCommit, lifecycle)
	if err != nil {
		t.Fatal(err)
	}
	source, err := json.Marshal(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	suite, err := CompileSuite(source, campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	result, err := campaign.Evaluate(ctx, suite)
	if failPublication {
		if !errors.Is(err, errPublicationFixture) || result.Report.Kind() != artifact.KindEvaluation ||
			len(result.Metrics) == 0 || result.Run.Valid() || result.Evidence.Valid() || result.Evaluation.Valid() {
			t.Fatalf("publication failure lost report or claimed completion: %+v, %v", result, err)
		}
		retained := result.Report
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		database, err = overgodb.Open(root)
		if err != nil {
			t.Fatal(err)
		}
		store = &publicationFaultStore{Store: database}
		campaign, err = newCampaign(store, runtime, identity, environment, runtimeLedgerCommit, lifecycle)
		if err != nil {
			t.Fatal(err)
		}
		result, err = campaign.Evaluate(ctx, suite)
		if result.Report != retained || runtime.generateCalls != len(exactFixture().Cases) {
			t.Fatalf("restart reacquired completed generation: report=%s, calls=%d", result.Report, runtime.generateCalls)
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := RequireEvaluationEvidence(ctx, store, result.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	wall, wallObserved := result.Resources.Measure(runrecord.ResourceWallNS)
	_, gpuObserved := result.Resources.Measure(runrecord.ResourceGPUNS)
	if !wallObserved || wall == 0 || gpuObserved ||
		result.Resources.Scope.Model != identity.Model || result.Resources.Scope.Hardware != environment.ID ||
		result.Resources.Scope.Workload != suite.Plan().Identity() || result.Resources.Scope.Attempt != result.Run {
		t.Fatalf("isolated resource evidence = %+v; bundle = %+v", result.Resources, evidence)
	}
	if lifecycle == LifecycleIsolated {
		if evidence.Resources == nil || evidence.ResourceObservation.Kind() != artifact.KindEvidence {
			t.Fatal("isolated resource proof absent")
		}
	} else {
		if evidence.Resources != nil || evidence.ResourceObservation.Valid() {
			t.Fatal("resident evaluation claimed isolated resources")
		}
		run, err := runrecord.RequireRun(ctx, store, result.Run)
		if err != nil {
			t.Fatal(err)
		}
		observationID, found, err := artifact.ResolveAlias(ctx, store, runrecord.ObservationChunkAlias(run.ID))
		if err != nil || !found {
			t.Fatalf("resident observation absent: %v", err)
		}
		observation, err := runrecord.RequireObservationChunkSummary(ctx, store, observationID)
		if err != nil || observation.Aggregate.Scope.Attempt != run.ID {
			t.Fatalf("resident observation differs: %+v, %v", observation, err)
		}
		record, err := runrecord.NewEvaluation(identity.Recipe, run.ID, suite.plan.Dataset(), result.Metrics)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := newEvaluationEvidence(ctx, store, suite.plan, suite.acceptance, suite.evaluator,
			result.Report, run, record, &observation); err == nil {
			t.Fatal("resident observation accepted as isolated proof")
		}
	}
	history, err := campaign.History(ctx, []CompiledSuite{suite}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.generateCalls != len(exactFixture().Cases) || len(history) != 1 {
		t.Fatalf("runtime calls/history = %d/%d", runtime.generateCalls, len(history))
	}
	entry := history[0]
	if entry.Run != result.Run || entry.Evaluation != result.Evaluation || entry.Report != result.Report ||
		entry.Recipe != identity.Recipe || entry.Dataset != suite.Descriptor().Dataset ||
		entry.Outcome != runrecord.OutcomeSucceeded || len(entry.Metrics) != len(result.Metrics) {
		t.Fatalf("indexed history = %+v; result = %+v", entry, result)
	}
}
