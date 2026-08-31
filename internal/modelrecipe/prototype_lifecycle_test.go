package modelrecipe

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/dataset"
	"overgo/internal/invocation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
)

const (
	verificationEvidenceMediaType = "application/vnd.overgo.test-candidate-evaluation+json"
	verificationEvidenceSchema    = "overgo/test-candidate-evaluation/v1"
)

type supervisedVerificationFixture struct {
	path       string
	store      *overgodb.Store
	definition recipe.Definition
	decision   recipe.Decision
	terminal   runrecord.StageReceipt
	running    runrecord.StageReceipt
	active     artifact.ID
	automation artifact.ID
}

// TestPrototypeRecipeLifecycleAndRollback proves the early lifecycle stops at
// a durable non-serving verified state. It also proves repository transaction
// rollback: a stale receipt CAS publishes neither the terminal receipt nor the
// lifecycle event.
func TestPrototypeRecipeLifecycleAndRollback(t *testing.T) {
	t.Run("verified and cold replayed", func(t *testing.T) {
		fixture := newSupervisedVerificationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		counting := &repositorytest.CountingRepository{Repository: fixture.store}
		_, event, err := Transition(
			t.Context(), counting, "fixture/supervised/verified", fixture.definition,
			recipe.StatusVerified, []artifact.ID{fixture.decision.ID}, nil, fixture.terminal,
		)
		if err != nil {
			t.Fatal(err)
		}
		if counting.Commits != 1 || event.To != recipe.StatusVerified {
			t.Fatalf("verified transition = (%d commits, %+v)", counting.Commits, event)
		}
		assertSupervisedAliases(t, fixture.store, fixture, event)
		if err := fixture.store.Close(); err != nil {
			t.Fatal(err)
		}
		reopened, err := overgodb.Open(fixture.path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = reopened.Close() })
		assertSupervisedAliases(t, reopened, fixture, event)
	})

	t.Run("stale receipt CAS rolls back lifecycle batch", func(t *testing.T) {
		fixture := newSupervisedVerificationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		decoy := testutil.ArtifactID(t, artifact.KindEvidence, "stale-stage-head")
		alias := runrecord.StageReceiptAliasRoot + fixture.terminal.Operation.String() + "/" + string(fixture.terminal.Node)
		conflicting := &conflictingLifecycleRepository{
			Repository: fixture.store,
			before: sync.OnceValue(func() error {
				_, err := fixture.store.Commit(t.Context(), artifact.Batch{
					Key:       "fixture/supervised/stale-stage",
					Artifacts: []artifact.Descriptor{{ID: decoy, Size: 1}},
					Aliases: []artifact.AliasBinding{{
						Name: alias, Target: decoy, Previous: artifact.IDPointer(fixture.running.ID),
					}},
				})
				return err
			}),
		}
		_, event, err := Transition(
			t.Context(), conflicting, "fixture/supervised/conflict", fixture.definition,
			recipe.StatusVerified, []artifact.ID{fixture.decision.ID}, nil, fixture.terminal,
		)
		if !errors.Is(err, overgodb.ErrAliasConflict) {
			t.Fatalf("stale receipt CAS = %v", err)
		}
		status, found, statusErr := Status(t.Context(), fixture.store, fixture.definition.ID)
		if statusErr != nil || !found || status != recipe.StatusValidated {
			t.Fatalf("lifecycle changed after rollback = (%s, %v, %v)", status, found, statusErr)
		}
		completed := fixture.terminal
		completed.Previous = fixture.running.ID
		completed, err = runrecord.NewStageReceipt(completed)
		if err != nil {
			t.Fatal(err)
		}
		for name, id := range map[string]artifact.ID{"event": event.ID, "terminal receipt": completed.ID} {
			if _, present, lookupErr := fixture.store.Artifact(t.Context(), id); lookupErr != nil || present {
				t.Fatalf("%s survived rolled-back batch = (%v, %v)", name, present, lookupErr)
			}
		}
	})
}

// TestTrainingLaunchUsesDirectOwnerWithCapabilityReceipt proves a training
// candidate cannot become launch-eligible through a document named evaluation
// or promotion. The direct internal invocation, effect, preflight, ceiling,
// operator decision, and admitted-running-terminal receipt chain are mandatory;
// no inward transport boundary can substitute for this owner.
func TestTrainingLaunchUsesDirectOwnerWithCapabilityReceipt(t *testing.T) {
	fixture := newSupervisedVerificationFixtureForTask(t, recipe.TaskTraining)
	t.Cleanup(func() { _ = fixture.store.Close() })
	ctx := t.Context()
	if fixture.definition.Task != recipe.TaskTraining || fixture.terminal.Invocation == nil ||
		fixture.terminal.Invocation.Boundary != invocation.BoundaryInternal {
		t.Fatalf("training launch fixture is not direct: %+v", fixture.terminal)
	}
	if _, _, err := Transition(
		ctx, fixture.store, "fixture/supervised/bypass", fixture.definition,
		recipe.StatusVerified, []artifact.ID{fixture.decision.ID}, nil,
	); err == nil {
		t.Fatal("raw lifecycle transition bypassed invocation receipt")
	}

	tests := map[string]func(*runrecord.StageReceipt){
		"invocation": func(value *runrecord.StageReceipt) { value.Invocation = nil },
		"boundary": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Boundary = invocation.BoundaryAgent
			value.Invocation = &binding
		},
		"kind": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Kind = invocation.MutationModelActivation
			value.Invocation = &binding
		},
		"action": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Action = string(invocation.MutationRollback)
			value.Invocation = &binding
		},
		"arguments": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Arguments = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-verification-arguments")
			value.Invocation = &binding
		},
		"effect": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Effect = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-verification-effect")
			value.Invocation = &binding
		},
		"preflight": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Preflight = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-verification-preflight")
			value.Invocation = &binding
		},
		"inspection": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Inspection = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-verification-inspection")
			value.Invocation = &binding
		},
		"ceiling": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Ceiling = testutil.ArtifactID(t, artifact.KindRecipe, "substituted-verification-ceiling")
			value.Invocation = &binding
		},
		"authority": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Authority = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-verification-authority")
			value.Invocation = &binding
		},
		"head": func(value *runrecord.StageReceipt) {
			binding := *value.Invocation
			binding.Head = artifact.CommitID{}
			value.Invocation = &binding
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			forged := cloneVerificationReceipt(fixture.terminal)
			mutate(&forged)
			if _, _, err := Transition(
				ctx, fixture.store, "fixture/supervised/refused/"+name, fixture.definition,
				recipe.StatusVerified, []artifact.ID{fixture.decision.ID}, nil, forged,
			); err == nil {
				t.Fatalf("verification accepted substituted %s", name)
			}
		})
	}
	if _, _, err := Transition(
		ctx, fixture.store, "fixture/supervised/accepted", fixture.definition,
		recipe.StatusVerified, []artifact.ID{fixture.decision.ID}, nil, fixture.terminal,
	); err != nil {
		t.Fatal(err)
	}
}

func newSupervisedVerificationFixture(t *testing.T) supervisedVerificationFixture {
	return newSupervisedVerificationFixtureForTask(t, recipe.TaskInference)
}

func newSupervisedVerificationFixtureForTask(t *testing.T, launchTask recipe.Task) supervisedVerificationFixture {
	t.Helper()
	ctx := t.Context()
	path := t.TempDir()
	store, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "supervised-verification-model")
	testutil.PublishArtifact(t, store, modelID)
	var definition recipe.Definition
	if launchTask == recipe.TaskTraining {
		definition, err = recipe.NewDefinitionWithDependencies(
			recipe.TaskTraining,
			[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
			[]recipe.Node{{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost}},
			nil, nil,
			[]recipe.Output{{
				Name: "batch", Data: recipe.DataBatch,
				Source: recipe.Endpoint{Node: "batch", Port: "batch"},
			}},
		)
	} else {
		definition, err = inferenceFixture(modelID, recipe.PlacementHost, DecodeSessionRequest)
	}
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/supervised/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(
		ctx, store, "fixture/supervised/validated", definition, recipe.StatusValidated, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	evaluation := publishVerificationFact(t, store, "candidate-evaluation")
	decision, err := recipe.NewDecision(
		definition.ID, recipe.DecisionAccepted, recipe.EvidenceVerified, "held-out candidate improvement accepted",
		recipe.Decider{CodeCommit: lifecycleDecisionCommit, Derivation: evaluation}, []artifact.ID{evaluation},
	)
	if err != nil {
		t.Fatal(err)
	}
	publishVerificationDecision(t, store, decision)
	task := publishVerificationTask(t, store, definition.ID, evaluation)
	arguments, err := verificationArguments(definition.ID, decision.ID, statusAlias(definition.ID))
	if err != nil {
		t.Fatal(err)
	}
	commitContent(t, store, "fixture/supervised/arguments", arguments, nil)
	mutation, err := invocation.NewEffect(invocation.Effect{
		Manual: definition.ID, Arguments: arguments.Descriptor.ID, Class: invocation.ClassMutation,
		Targets: []invocation.Target{{Scope: invocation.ScopeRepository, Value: statusAlias(definition.ID)}}, Known: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	publishInvocationEffect(t, store, "fixture/supervised/mutation", mutation)
	inspectionFact := publishVerificationFact(t, store, "candidate-status-inspection")
	inspectionEffect, err := invocation.NewEffect(invocation.Effect{
		Manual: definition.ID, Arguments: arguments.Descriptor.ID, Class: invocation.ClassInspection,
		Targets: []invocation.Target{{Scope: invocation.ScopeRepository, Value: statusAlias(definition.ID)}}, Known: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	publishInvocationEffect(t, store, "fixture/supervised/inspection-effect", inspectionEffect)
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "supervised-verification-operation")
	testutil.PublishArtifact(t, store, operation)
	head, _ := store.Head()
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: 1, MaxBytes: 1, MaxDocuments: 1, MaxDepth: 1, MaxResults: 1,
	}, head, []dataset.InteractionSelectionSource{{
		Source: arguments.Descriptor.ID, CausalRoot: operation,
		Tokens: 1, Bytes: 1, Documents: 1, Depth: 1,
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	stimulus, err := runrecord.PublishAttemptStimulus(ctx, store, runrecord.AttemptStimulusBoundary{
		Operation: operation, Attempt: 1, Manual: definition.ID, Class: invocation.ClassMutation,
		Arguments: arguments.Descriptor.ID, Effect: mutation.ID,
		Inspection: inspectionFact, InspectionEffect: inspectionEffect.ID,
		Ceiling: task.ID, CausalContext: inspectionFact, Prior: inspectionFact, Selection: selection,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	action := operatoraction.Action{
		Code: string(invocation.MutationPromotion), Summary: string(invocation.MutationPromotion),
		Argv: invocation.ApprovalArguments(definition.ID, arguments.Descriptor.ID, mutation.ID, stimulus.ID),
	}
	request, err := operatoraction.NewApprovalRequest(operation, task.ID, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	human, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.PublishHumanDecision(ctx, store, request, human); err != nil {
		t.Fatal(err)
	}
	binding := invocation.ReceiptBinding{
		Boundary: invocation.BoundaryInternal, Kind: invocation.MutationPromotion, Action: string(invocation.MutationPromotion),
		Subject: definition.ID, Arguments: arguments.Descriptor.ID, Effect: mutation.ID,
		Preflight: stimulus.ID, Inspection: inspectionFact, Ceiling: task.ID, Authority: human.ID,
		CausalContext: inspectionFact, Head: selection.Head,
	}
	input := []runrecord.StageBinding{{
		Port: recipe.PortName(recipe.StatusVerified), Artifacts: []artifact.ID{decision.ID},
	}}
	admitted, err := runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
		Recipe: task.ID, Node: recipe.NodeID(invocation.MutationPromotion), Operation: operation,
		Attempt: 1, State: runrecord.StageAdmitted, Inputs: input, Invocation: &binding,
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	running, err := runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
		Recipe: task.ID, Node: recipe.NodeID(invocation.MutationPromotion), Operation: operation,
		Attempt: 1, State: runrecord.StageRunning, Inputs: input, Invocation: &binding,
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if running.Previous != admitted.ID {
		t.Fatalf("running receipt previous = %s, want %s", running.Previous, admitted.ID)
	}
	active := testutil.ArtifactID(t, artifact.KindRecipe, "unchanged-production-incumbent")
	automation := testutil.ArtifactID(t, artifact.KindProfile, "unchanged-automation-grant")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/supervised/incumbents",
		Artifacts: []artifact.Descriptor{{ID: active, Size: 1}, {ID: automation, Size: 1}},
		Aliases: []artifact.AliasBinding{
			{Name: activeAlias(definition.Model, definition.Task), Target: active},
			{Name: runrecord.AutomationPolicyActiveAliasRoot + "supervised-fixture", Target: automation},
		},
	}); err != nil {
		t.Fatal(err)
	}
	return supervisedVerificationFixture{
		path: path, store: store, definition: definition, decision: decision,
		terminal: runrecord.StageReceipt{
			Recipe: task.ID, Node: recipe.NodeID(invocation.MutationPromotion), Operation: operation,
			Attempt: 1, State: runrecord.StageCompleted, Inputs: input, Invocation: &binding,
		},
		running: running, active: active, automation: automation,
	}
}

func publishVerificationFact(t *testing.T, store artifact.Repository, name string) artifact.ID {
	t.Helper()
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: verificationEvidenceMediaType, Schema: verificationEvidenceSchema,
	}
	content, err := contract.ContentBytes([]byte(`{"passed":true,"subject":"` + name + `"}`))
	if err != nil {
		t.Fatal(err)
	}
	commitContent(t, store, "fixture/supervised/fact/"+name, content, nil)
	return content.Descriptor.ID
}

func publishVerificationDecision(t *testing.T, store artifact.Repository, decision recipe.Decision) {
	t.Helper()
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitContent(t, store, "fixture/supervised/decision", content, decision.Lineage())
}

func publishVerificationTask(
	t *testing.T,
	store artifact.Repository,
	agent, evidence artifact.ID,
) recipe.AgentTaskContract {
	t.Helper()
	budget := publishVerificationFact(t, store, "candidate-verification-budget")
	verifier := publishVerificationFact(t, store, "candidate-verification-criterion")
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: agent, Objective: "Verify one candidate without serving it.",
		Scope: []string{statusAlias(agent)}, AllowedEffects: []string{string(invocation.MutationPromotion)},
		Acceptance: []recipe.AcceptanceCriterion{{
			Name: "candidate-verification", Scope: "Held-out evaluation is accepted.", Verifier: verifier,
		}},
		Verification: []artifact.ID{evidence}, Budget: budget,
		PauseConditions: []string{"Operator declines the exact mutation."},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := task.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitContent(t, store, "fixture/supervised/task", content, task.Lineage())
	return task
}

func publishInvocationEffect(t *testing.T, store artifact.Repository, key string, effect invocation.Effect) {
	t.Helper()
	content, err := effect.Content()
	if err != nil {
		t.Fatal(err)
	}
	commitContent(t, store, key, content, effect.Lineage())
}

func commitContent(
	t *testing.T,
	store artifact.Repository,
	key string,
	content artifact.Content,
	lineage []artifact.Lineage,
) {
	t.Helper()
	batch, err := artifact.NewDocumentBatch(key, []artifact.Content{content}, lineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
}

func assertSupervisedAliases(
	t *testing.T,
	reader artifact.Reader,
	fixture supervisedVerificationFixture,
	event recipe.LifecycleEvent,
) {
	t.Helper()
	status, found, err := Status(t.Context(), reader, fixture.definition.ID)
	if err != nil || !found || status != recipe.StatusVerified {
		t.Fatalf("verified status = (%s, %v, %v)", status, found, err)
	}
	active, found, err := reader.ResolveAlias(t.Context(), activeAlias(fixture.definition.Model, fixture.definition.Task))
	if err != nil || !found || active != fixture.active {
		t.Fatalf("production incumbent changed = (%s, %v, %v)", active, found, err)
	}
	automation, found, err := reader.ResolveAlias(t.Context(), runrecord.AutomationPolicyActiveAliasRoot+"supervised-fixture")
	if err != nil || !found || automation != fixture.automation {
		t.Fatalf("automation grant changed = (%s, %v, %v)", automation, found, err)
	}
	receipt, found, err := runrecord.ResolveStageReceipt(
		t.Context(), reader, fixture.terminal.Operation, fixture.terminal.Node,
	)
	if err != nil || !found || receipt.State != runrecord.StageCompleted ||
		!slices.Contains(event.Evidence, receipt.ID) {
		t.Fatalf("terminal receipt = (%+v, %v, %v); evidence=%v", receipt, found, err, event.Evidence)
	}
}

func cloneVerificationReceipt(value runrecord.StageReceipt) runrecord.StageReceipt {
	value.Inputs = slices.Clone(value.Inputs)
	for index := range value.Inputs {
		value.Inputs[index].Artifacts = slices.Clone(value.Inputs[index].Artifacts)
	}
	if value.Invocation != nil {
		binding := *value.Invocation
		value.Invocation = &binding
	}
	return value
}

type conflictingLifecycleRepository struct {
	artifact.Repository
	before func() error
}

func (repository *conflictingLifecycleRepository) Commit(
	ctx context.Context,
	batch artifact.Batch,
) (artifact.CommitID, error) {
	if err := repository.before(); err != nil {
		return artifact.CommitID{}, err
	}
	return repository.Repository.Commit(ctx, batch)
}
