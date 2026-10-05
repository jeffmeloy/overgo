package evaluation

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestEvaluationAtAnUnchangedPlanReusesItsCampaign evaluates one suite twice at one commit, as a
// cycle rerun at an unchanged HEAD does: the second evaluation is not refused where the first
// bound the plan's campaign, measures nothing again, and leaves the campaign bound to the first.
func TestEvaluationAtAnUnchangedPlanReusesItsCampaign(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	identity := modelrecipe.ProgramIdentity{
		Model:      planID(t, artifact.KindModel, "rerun model"),
		Definition: planID(t, artifact.KindModelDefinition, "rerun definition"),
		Recipe:     planID(t, artifact.KindRecipe, "rerun recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/rerun", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &observedRuntime{}
	campaign, err := newCampaign(store, runtime, identity, environment, runtimeLedgerCommit, LifecycleIsolated)
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
	first, err := campaign.Evaluate(ctx, suite)
	if err != nil {
		t.Fatal(err)
	}
	calls := runtime.generateCalls
	again, err := campaign.Evaluate(ctx, suite)
	if err != nil {
		t.Fatalf("a second evaluation of the same plan at the same commit: %v", err)
	}
	if runtime.generateCalls != calls || again.Report != first.Report {
		t.Errorf("the second evaluation generated %d more times and reported %s, want it to reuse the first's %s", runtime.generateCalls-calls, again.Report, first.Report)
	}
	bound, found, err := artifact.ResolveAlias(ctx, store, campaignAlias(suite.Plan().Identity()))
	if err != nil || !found || bound != first.Report {
		t.Errorf("the plan's campaign is bound to %s (found %t, %v), want the first evaluation's report %s", bound, found, err, first.Report)
	}
}
