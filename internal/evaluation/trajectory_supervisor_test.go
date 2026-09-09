package evaluation

import (
	"errors"
	"slices"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestTrajectorySupervisorStopsExactNoProgress(t *testing.T) {
	fixture := newTrajectorySupervisorFixture(t)
	previous := fixture.exchange(t, "previous", `{}`, nil, nil, nil)
	current := fixture.exchange(t, "current", `{}`, nil, nil, nil)
	decision, commit, err := SuperviseTrajectory(t.Context(), fixture.store, previous.ID, current.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !commit.Valid() || decision.Decision != executionfailure.DecisionRetireSession ||
		decision.Rule != trajectoryRuleExactNoProgress || len(decision.Progress) != 0 {
		t.Fatalf("decision = %+v, want exact no-progress retirement", decision)
	}
	loaded, err := RequireTrajectoryHealthDecision(t.Context(), fixture.store, decision.ID)
	if err != nil || loaded.ID != decision.ID {
		t.Fatalf("loaded decision = %+v, %v", loaded, err)
	}
}

func TestTrajectorySupervisorPermitsNewEvidence(t *testing.T) {
	fixture := newTrajectorySupervisorFixture(t)
	previousBase := fixture.baseExchange(t, "previous", `{}`)
	currentBase := fixture.baseExchange(t, "current", `{}`)
	obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
		Task: fixture.recipe, Name: "verify-result", Scope: "trajectory", MutationEpoch: 1,
		Sources: []artifact.ID{previousBase.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	resolution, err := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{
		Obligation: obligation.ID, Scope: obligation.Scope, MutationEpoch: obligation.MutationEpoch,
		Evidence: []artifact.ID{currentBase.receipt, currentBase.result},
	})
	if err != nil {
		t.Fatal(err)
	}
	obligationContent, _ := obligation.Content()
	resolutionContent, _ := resolution.Content()
	batch, err := artifact.NewDocumentBatch(
		"trajectory/obligation", []artifact.Content{obligationContent, resolutionContent},
		append(obligation.Lineage(), resolution.Lineage()...), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	previous := fixture.agentTrajectory(t, previousBase.trace, []artifact.ID{obligation.ID}, nil, nil)
	current := fixture.agentTrajectory(t, currentBase.trace, []artifact.ID{obligation.ID}, []artifact.ID{resolution.ID}, nil)
	decision, _, err := SuperviseTrajectory(t.Context(), fixture.store, previous.ID, current.ID)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Decision != executionfailure.DecisionResume || decision.Rule != trajectoryRuleNewEvidence ||
		len(decision.Progress) != 1 || decision.Progress[0] != resolution.ID {
		t.Fatalf("decision = %+v, want continuation on typed resolution", decision)
	}
}

func TestTrajectorySupervisorIgnoresUnrelatedFreshIDs(t *testing.T) {
	fixture := newTrajectorySupervisorFixture(t)
	previous := fixture.exchange(t, "previous", `{}`, nil, nil, nil)
	unrelated := testutil.ArtifactID(t, artifact.KindEvidence, "unrelated-trajectory-id")
	testutil.PublishArtifact(t, fixture.store, unrelated)
	current := fixture.exchange(t, "current", `{}`, nil, nil, []artifact.ID{unrelated})
	decision, _, err := SuperviseTrajectory(t.Context(), fixture.store, previous.ID, current.ID)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Decision != executionfailure.DecisionRetireSession || decision.Rule != trajectoryRuleExactNoProgress {
		t.Fatalf("unrelated ID changed no-progress decision: %+v", decision)
	}
}

func TestTrajectorySupervisorRejectsSyntheticTrace(t *testing.T) {
	fixture := newTrajectorySupervisorFixture(t)
	previous := fixture.exchange(t, "previous", `{}`, nil, nil, nil)
	synthetic := testutil.ArtifactID(t, artifact.KindEvidence, "synthetic-trajectory")
	testutil.PublishArtifact(t, fixture.store, synthetic)
	if _, _, err := SuperviseTrajectory(t.Context(), fixture.store, previous.ID, synthetic); err == nil {
		t.Fatal("descriptor-only trajectory was admitted")
	}
}

func TestRequireTrajectoryHealthDecisionRejectsForgedLineage(t *testing.T) {
	fixture := newTrajectorySupervisorFixture(t)
	previousBase := fixture.baseExchange(t, "previous", `{}`)
	currentBase := fixture.baseExchange(t, "current", `{}`)
	previous := fixture.agentTrajectory(t, previousBase.trace, nil, nil, nil)
	current := fixture.agentTrajectory(t, currentBase.trace, nil, nil, nil)
	decision, err := trajectoryHealthDecisionCodec.New(TrajectoryHealthDecision{
		Version:         artifact.InitialDocumentVersion,
		Previous:        previous.ID,
		Current:         current.ID,
		PreviousReceipt: previousBase.receipt,
		CurrentReceipt:  currentBase.receipt,
		Decision:        executionfailure.DecisionRetireSession,
		Rule:            trajectoryRuleExactNoProgress,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := trajectoryHealthDecisionCodec.Content(decision)
	if err != nil {
		t.Fatal(err)
	}
	forgedLineage := slices.DeleteFunc(decision.Lineage(), func(edge artifact.Lineage) bool {
		return edge.Parent == decision.CurrentReceipt
	})
	batch, err := artifact.NewDocumentBatch(
		"trajectory/health/forged-lineage", []artifact.Content{content}, forgedLineage, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
		t.Fatal(err)
	}
	if _, err := trajectoryHealthDecisionCodec.Require(t.Context(), fixture.store, decision.ID); err != nil {
		t.Fatalf("valid decision body was not stored: %v", err)
	}
	if _, err := RequireTrajectoryHealthDecision(t.Context(), fixture.store, decision.ID); err == nil {
		t.Fatal("decision detached from its exact lineage was admitted")
	}
}

type trajectorySupervisorFixture struct {
	store  *overgodb.Store
	manual agenttool.Manual
	recipe artifact.ID
	model  artifact.ID
}

type trajectoryBase struct {
	ID      artifact.ID
	trace   runrecord.InteractionTrace
	receipt artifact.ID
	result  artifact.ID
}

func newTrajectorySupervisorFixture(t *testing.T) trajectorySupervisorFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return newTrajectorySupervisorFixtureInStore(t, store)
}

func newTrajectorySupervisorFixtureInStore(t *testing.T, store *overgodb.Store) trajectorySupervisorFixture {
	t.Helper()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "trajectory-recipe")
	model := testutil.ArtifactID(t, artifact.KindModel, "trajectory-model")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "trajectory/authorities", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: model}},
	}); err != nil {
		t.Fatal(err)
	}
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "trajectory.inspect", Description: "Inspect exact trajectory state.",
		Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	return trajectorySupervisorFixture{store: store, manual: manual, recipe: recipeID, model: model}
}

func (fixture trajectorySupervisorFixture) exchange(
	t *testing.T,
	name, arguments string,
	obligations, resolutions, context []artifact.ID,
) runrecord.InteractionTrace {
	t.Helper()
	base := fixture.baseExchange(t, name, arguments)
	return fixture.agentTrajectory(t, base.trace, obligations, resolutions, context)
}

func (fixture trajectorySupervisorFixture) baseExchange(t *testing.T, name, arguments string) trajectoryBase {
	t.Helper()
	ctx := t.Context()
	argumentContent, err := runrecord.AttemptArgumentContent([]byte(arguments))
	if err != nil {
		t.Fatal(err)
	}
	argumentBatch, err := artifact.NewDocumentBatch("trajectory/arguments/"+name, []artifact.Content{argumentContent}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, fixture.store, argumentBatch); err != nil && !errorsIsNoChange(err) {
		t.Fatal(err)
	}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory-operation-"+name)
	resultText := "stable-result"
	result, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(resultText))
	if err != nil {
		t.Fatal(err)
	}
	base := runrecord.StageReceipt{
		Recipe: fixture.recipe, Node: recipe.NodeID("tool"), Operation: operation, Attempt: 1,
		State:  runrecord.StageAdmitted,
		Inputs: []runrecord.StageBinding{{Port: "input", Artifacts: []artifact.ID{fixture.manual.ID, argumentContent.Descriptor.ID}}},
	}
	if _, err := runrecord.PublishStageReceipt(ctx, fixture.store, base, nil, []artifact.Descriptor{{ID: operation}}); err != nil {
		t.Fatal(err)
	}
	base.State = runrecord.StageRunning
	if _, err := runrecord.PublishStageReceipt(ctx, fixture.store, base, nil, nil); err != nil {
		t.Fatal(err)
	}
	base.State = runrecord.StageCompleted
	base.Outputs = []runrecord.StageBinding{{Port: "output", Artifacts: []artifact.ID{result}}}
	receipt, err := runrecord.PublishStageReceipt(ctx, fixture.store, base, nil, []artifact.Descriptor{{ID: result, Size: uint64(len(resultText))}})
	if err != nil {
		t.Fatal(err)
	}
	interaction, err := runrecord.PublishInteraction(ctx, fixture.store, runrecord.Interaction{
		Response: "trajectory-" + name, Recipe: fixture.recipe, Model: fixture.model,
		Node: recipe.NodeID("tool"), Operation: operation, Run: receipt.ID,
	}, []runrecord.InteractionMessage{
		{Role: "user", Content: "inspect trajectory state"},
		{Role: "assistant", ToolCalls: []runrecord.InteractionToolCall{{
			ID: "call", Type: "function", Name: fixture.manual.Name, Manual: fixture.manual.ID, Arguments: arguments,
		}}},
		{Role: "tool", Content: resultText, ToolCallID: "call"},
		{Role: "assistant", Content: "done"},
	}, "")
	if err != nil {
		t.Fatal(err)
	}
	trace, err := runrecord.RequireInteractionTrace(ctx, fixture.store, interaction.Trace)
	if err != nil {
		t.Fatal(err)
	}
	return trajectoryBase{ID: trace.ID, trace: trace, receipt: receipt.ID, result: result}
}

func (fixture trajectorySupervisorFixture) agentTrajectory(
	t *testing.T,
	base runrecord.InteractionTrace,
	obligations, resolutions, context []artifact.ID,
) runrecord.InteractionTrace {
	t.Helper()
	base.TaskContract, base.Terminal = fixture.recipe, runrecord.OutcomeSucceeded
	base.ToolManuals = []artifact.ID{fixture.manual.ID}
	base.Obligations, base.Resolutions, base.Context = obligations, resolutions, context
	trajectory, err := runrecord.NewAgentTrajectory(base)
	if err != nil {
		t.Fatal(err)
	}
	content, err := trajectory.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"trajectory/agent/"+trajectory.ID.String(), []artifact.Content{content}, trajectory.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil && !errorsIsNoChange(err) {
		t.Fatal(err)
	}
	return trajectory
}

func errorsIsNoChange(err error) bool { return errors.Is(err, artifact.ErrNoChange) }
