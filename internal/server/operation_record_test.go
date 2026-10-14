package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// operationRecordFamilies are the alias roots a record under an operation's id may have.
//   - each root is one a production reader finds again by the operation's id
//   - the operation's views read all four (Handler.operationEvidenceSnapshot)
//   - the workflow replay reads the stage receipts as well (workflowruntime.Runtime.recoverStage)
var operationRecordFamilies = []string{
	runrecord.StageReceiptAliasRoot,
	runrecord.ServingAttemptAliasRoot,
	runrecord.InteractionOperationAliasRoot,
	runrecord.HumanDecisionAliasRoot,
}

// recordWorkflow is a workflow workspace that completes with one run.
//   - unheld set = the execution commits no run, so its completion names a run the store lacks
//   - block set = the next execution ends blocked on the operator, with the run it committed
type recordWorkflow struct {
	store      artifact.Repository
	capability WorkflowCapability
	run        artifact.ID
	resume     operatoraction.Action
	unheld     atomic.Bool
	block      atomic.Bool
}

// newRecordWorkflow makes a generation workflow of one text control, whose run and recipe are named for name.
func newRecordWorkflow(t *testing.T, store artifact.Repository, name string) *recordWorkflow {
	t.Helper()
	return &recordWorkflow{
		store: store, run: testutil.ArtifactID(t, artifact.KindRun, name+" run"),
		resume: operatoraction.Action{Code: "resume", Summary: "Resume the blocked work", Argv: []string{"overgo", "resume"}},
		capability: WorkflowCapability{
			Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name+" recipe"),
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{{Name: "text", Type: workflowControlText, Required: true}},
		},
	}
}

// request is one submission of the workflow, numbered when number is not empty.
func (workflow *recordWorkflow) request(number string) workflowRequest {
	return workflowRequest{Task: workflow.capability.Task, Recipe: workflow.capability.Recipe, Input: json.RawMessage(`{"text":"one request"}`), Request: number}
}

func (workflow *recordWorkflow) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if kind != workflowGeneration {
		return nil, nil
	}
	return []WorkflowCapability{workflow.capability}, nil
}

func (workflow *recordWorkflow) ExecuteWorkflow(ctx context.Context, _ WorkflowKind, capability WorkflowCapability, _ json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	completion := operation.Completion{Run: workflow.run}
	if workflow.unheld.Load() {
		return completion, nil
	}
	if _, err := workflow.store.Commit(ctx, artifact.Batch{
		Key: "fixture/operation-record/" + workflow.run.String(), Artifacts: []artifact.Descriptor{{ID: workflow.run}},
	}); err != nil {
		return completion, err
	}
	if workflow.block.CompareAndSwap(true, false) {
		return completion, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
			Subject: capability.Recipe, Reason: "operator approval required", Evidence: []artifact.ID{},
			Actions: []operatoraction.Action{workflow.resume},
		})
	}
	reporter.Publishing()
	return completion, nil
}

// countedWorkflow counts the executions a workflow workspace is asked for.
type countedWorkflow struct {
	WorkflowWorkspaceAPI
	executions atomic.Int32
}

func (workflow *countedWorkflow) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	workflow.executions.Add(1)
	return workflow.WorkflowWorkspaceAPI.ExecuteWorkflow(ctx, kind, capability, raw, reporter)
}

// recordRuntime answers the one case of the test's exact suite.
type recordRuntime struct{}

func (recordRuntime) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	ids := []tokenizer.TokenID{0, 1}
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: len(ids)})
	for index, piece := range []string{"o", "k"} {
		id := tokenizer.TokenID(index)
		ids = append(ids, id)
		if err := options.OnToken(inference.TokenEvent{ID: id, Piece: piece}); err != nil {
			return nil, "", err
		}
	}
	return ids, "", nil
}

func (recordRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("the operation record fixture scores no continuation")
}

// TestAnOperationsRecordIsFoundAgainOrIsNotWritten holds that an operation leaves no record under its own id that nothing finds again.
//   - entry = a production path into an operation manager, driven twice with one request on one temporary store
//   - record under an id = an alias whose name holds the id, since an alias is what a later reader finds a record by
//   - found again = the alias lies under a root of operationRecordFamilies, each of which names its reader
//   - an entry's second request runs under an id of its own, so it can find nothing the first left under its id
//   - the id is presented a second time by a granted recovery alone, whose first attempt ended in an error
//   - a numbered repeat is answered from the run ledger's memory with no second submission, and a later server has no such memory
//   - the library validation's executor is a stub here, because the real one loads a model, and its submission is the server's own
//   - the grounded replay's request names artifacts the store lacks, so both of its operations fail after they were submitted
//   - the tool executor is the one a server makes when its configuration holds a tool adapter, which no command sets
func TestAnOperationsRecordIsFoundAgainOrIsNotWritten(t *testing.T) {
	t.Parallel()
	const retention = 4
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, "operation record "+name)
	}

	// Two workflow workspaces of the test, and the model builder's own.
	plain, blocking := newRecordWorkflow(t, store, "operation record plain"), newRecordWorkflow(t, store, "operation record blocking")
	if _, err := scratchmodel.PublishDerivationProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	builder, err := NewModelBuilderWorkspace(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	plainRuns, blockingRuns, builds := &countedWorkflow{WorkflowWorkspaceAPI: plain}, &countedWorkflow{WorkflowWorkspaceAPI: blocking}, &countedWorkflow{WorkflowWorkspaceAPI: builder}

	// The evaluation workspace is the server's own, over a runtime that answers one exact case.
	model, definition, evaluated := id(artifact.KindModel, "model"), id(artifact.KindModelDefinition, "definition"), id(artifact.KindRecipe, "evaluation recipe")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/operation-record/evaluated", Artifacts: []artifact.Descriptor{{ID: model}, {ID: definition}, {ID: evaluated}},
	}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	suite, err := json.Marshal(evaluation.ExactSuite{Schema: "fixture/exact", Source: "operation-record", Cases: []evaluation.ExactCase{
		{Name: "one", Prompt: "fixture input", MaxTokens: 2, Text: "ok", PromptTokens: 2, GeneratedTokens: 2},
	}})
	if err != nil {
		t.Fatal(err)
	}
	suitePath := filepath.Join(t.TempDir(), "suite.json")
	if err := os.WriteFile(suitePath, suite, 0o600); err != nil {
		t.Fatal(err)
	}
	evaluations, err := NewEvaluationWorkspace(store, recordRuntime{}, modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: evaluated},
		environment, evaluationTestCommit, retention, []string{suitePath})
	if err != nil {
		t.Fatal(err)
	}

	// The library intake's validation is a stub that completes with a run it committed.
	var validations atomic.Int32
	validated, validationRun := id(artifact.KindRecipe, "validated recipe"), id(artifact.KindRun, "validation run")
	intake := LibraryIntake{
		ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
		Register: func(context.Context, *overgodb.Store, string, string) (map[string]any, error) {
			return nil, errors.New("the operation record fixture registers no model")
		},
		Validate: func(context.Context, *overgodb.Store, string, string) (artifact.ID, operation.Executor, error) {
			return validated, func(ctx context.Context, _ operation.Reporter) (operation.Completion, error) {
				validations.Add(1)
				_, err := store.Commit(ctx, artifact.Batch{Key: "fixture/operation-record/validation", Artifacts: []artifact.Descriptor{{ID: validationRun}}})
				return operation.Completion{Run: validationRun}, err
			}, nil
		},
	}

	// The tool executor is the one the server makes for a tool adapter, with a manager of its own.
	const toolModule recipe.ModuleID = "echo"
	toolModel := id(artifact.KindModel, "tool model")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/operation-record/tool", Artifacts: []artifact.Descriptor{{ID: toolModel}}}); err != nil {
		t.Fatal(err)
	}
	module := recipe.ToolModule(toolModule, recipe.PlacementHost)
	node := recipe.Node{ID: "execute", Module: module.ID, Placement: recipe.PlacementHost}
	toolDefinition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: toolModel}}, []recipe.Node{node}, nil,
		[]recipe.Input{{Name: recipe.ToolCallPort, Data: recipe.DataToolCall, Target: recipe.Endpoint{Node: node.ID, Port: recipe.ToolCallPort}}},
		[]recipe.Output{{Name: recipe.ToolResultPort, Data: recipe.DataToolResult, Source: recipe.Endpoint{Node: node.ID, Port: recipe.ToolResultPort}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := recipe.NewCatalog(module)
	if err != nil {
		t.Fatal(err)
	}
	toolProgram, err := recipe.CompileProgram(toolDefinition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	var toolCalls atomic.Int32
	config := Config{
		Repository: store, MaxStoredResponses: retention, Evaluation: evaluations, LibraryIntake: intake,
		Workflows:   WorkflowWorkspaceSet{plainRuns, blockingRuns, builds},
		ToolProgram: toolProgram,
		ToolAdapter: func(_ context.Context, call recipe.ToolCall) (recipe.ToolResult, error) {
			toolCalls.Add(1)
			return recipe.ToolResult{CallID: call.ID, Module: call.Module, Output: json.RawMessage(`{}`)}, nil
		},
	}
	serve := func() *Handler {
		t.Helper()
		handler, err := New(config, &fakeGenerator{})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handler.Close() })
		return handler
	}
	server := serve()

	// submit posts a request that starts an operation, and ended waits for the operation's last state.
	submit := func(handler *Handler, path string, body any) artifact.ID {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, path, marshalTestJSON(t, body))
		var submission workflowResponse
		if err := json.Unmarshal(response.Body.Bytes(), &submission); response.Code != http.StatusAccepted || err != nil || !submission.Operation.Valid() {
			t.Fatalf("%s answered %d %s (%v), want an operation", path, response.Code, response.Body.String(), err)
		}
		return submission.Operation
	}
	ended := func(handler *Handler, operationID artifact.ID) operation.State {
		t.Helper()
		status, err := handler.operations.Wait(ctx, operationID)
		if err != nil {
			t.Fatal(err)
		}
		return status.State
	}
	// driven names the entry of every operation the test started.
	driven := map[artifact.ID]string{}
	// twice drives one entry two times with one request, and holds each to the state the entry ends in.
	twice := func(entry string, handler *Handler, path string, body any, want operation.State) (first, second artifact.ID) {
		t.Helper()
		operations := [2]artifact.ID{}
		for index := range operations {
			operations[index] = submit(handler, path, body)
			if state := ended(handler, operations[index]); state != want {
				t.Fatalf("%s: request %d ended %s, want %s", entry, index, state, want)
			}
			driven[operations[index]] = entry
		}
		t.Logf("%s: the first request ran as %s and the second as %s", entry, operations[0], operations[1])
		return operations[0], operations[1]
	}

	// A workflow with no number runs each submission as an operation of its own, and each executes.
	if first, second := twice("workflow", server, "/generation/run", plain.request(""), operation.StateCompleted); first == second || plainRuns.executions.Load() != 2 {
		t.Errorf("workflow: two submissions ran as %s and %s with %d executions, want two operations that each executed", first, second, plainRuns.executions.Load())
	}

	// A numbered workflow's repeat is the first's operation by the ledger's memory, and a later server starts another.
	if first, second := twice("numbered workflow", server, "/generation/run", plain.request("run-one"), operation.StateCompleted); first != second || plainRuns.executions.Load() != 3 {
		t.Errorf("numbered workflow: a repeat ran as %s after %s with %d executions in all, want the one operation and three", second, first, plainRuns.executions.Load())
	}
	later := serve()
	restarted := submit(later, "/generation/run", plain.request("run-one"))
	if state := ended(later, restarted); state != operation.StateCompleted || driven[restarted] != "" || plainRuns.executions.Load() != 4 {
		t.Errorf("numbered workflow on a later server: ran as %s and ended %s with %d executions in all, want an operation of its own that executed", restarted, state, plainRuns.executions.Load())
	}
	t.Logf("numbered workflow on a later server: the request ran as %s", restarted)
	driven[restarted] = "numbered workflow on a later server"

	// A model build runs its stages under the manager's id, which is another id for the same corpus, seed and steps.
	build, err := builder.WorkflowCapabilities(ctx, workflowModelBuild)
	if err != nil || len(build) != 1 {
		t.Fatalf("model builder lists %d capabilities (%v), want one", len(build), err)
	}
	buildInput, err := json.Marshal(modelBuildInput{Corpus: "abcd\nbcda\ncdab\ndabc", Seed: 1, Steps: 1})
	if err != nil {
		t.Fatal(err)
	}
	if first, second := twice("model build", server, "/model-builder/run", workflowRequest{Task: build[0].Task, Recipe: build[0].Recipe, Input: buildInput}, operation.StateCompleted); first == second || builds.executions.Load() != 2 {
		t.Errorf("model build: two submissions ran as %s and %s with %d executions, want two operations that each executed", first, second, builds.executions.Load())
	}

	// An evaluation of plans, and a grounded replay.
	capabilities, err := evaluations.EvaluationCapabilities(ctx)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("evaluation workspace lists %d capabilities (%v), want one", len(capabilities), err)
	}
	if first, second := twice("evaluation", server, "/evaluations/run", evaluationRunRequest{Model: model, Plans: []artifact.ID{capabilities[0].Suite.Plan}}, operation.StateCompleted); first == second {
		t.Errorf("evaluation: two submissions ran as the one operation %s, want two", first)
	}
	var replay any
	if err := json.Unmarshal(groundedReplayAPIBody(t, model, groundedReplayAPIRequest(t, evaluated), false), &replay); err != nil {
		t.Fatal(err)
	}
	if first, second := twice("grounded replay", server, "/evaluations/run", replay, operation.StateFailed); first == second {
		t.Errorf("grounded replay: two submissions ran as the one operation %s, want two", first)
	}

	// A library validation, which answers with its operation as a workflow does.
	if first, second := twice("library validation", server, "/library/validate", libraryValidateRequest{Path: "model.gguf"}, operation.StateCompleted); first == second || validations.Load() != 2 {
		t.Errorf("library validation: two submissions ran as %s and %s with %d executions, want two operations that each executed", first, second, validations.Load())
	}

	// A granted recovery, the one entry that presents an id a second time, after an attempt that ended in an error.
	blocking.block.Store(true)
	blocked := submit(server, "/generation/run", blocking.request(""))
	if state := ended(server, blocked); state != operation.StateBlocked {
		t.Fatalf("granted recovery: the first attempt ended %s, want %s", state, operation.StateBlocked)
	}
	approval, err := operatoraction.NewApprovalRequest(blocked, blocking.capability.Recipe, blocking.resume, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	granted := serveTestRequest(server, http.MethodPost, "/operations/decision", marshalTestJSON(t, operationDecisionRequest{
		Operation: blocked, Tool: blocking.resume.Code, Answer: operatoraction.AnswerGrant, Request: approval.ID,
	}))
	var outcome operationDecisionResponse
	if err := json.Unmarshal(granted.Body.Bytes(), &outcome); granted.Code != http.StatusAccepted || err != nil || outcome.Operation != blocked {
		t.Fatalf("granted recovery: the decision answered %d %s (%v), want the operation %s again", granted.Code, granted.Body.String(), err, blocked)
	}
	if state := ended(server, blocked); state != operation.StateCompleted || blockingRuns.executions.Load() != 2 {
		t.Errorf("granted recovery: the second attempt ended %s after %d executions, want it completed by a second execution", state, blockingRuns.executions.Load())
	}
	t.Logf("granted recovery: both attempts ran as %s, and the second executed", blocked)
	driven[blocked] = "granted recovery"

	// A tool call is submitted to the executor's own manager, and its operations are the ones its stage receipts name.
	call, err := recipe.NewToolCall("call", toolModule, json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := server.tools.ExecuteTool(ctx, call); err != nil {
			t.Fatal(err)
		}
	}
	receipts := map[artifact.ID]int{}
	if err := store.VisitAliases(ctx, runrecord.StageReceiptAliasRoot, func(alias overgodb.AliasView) error {
		name, _, _ := strings.Cut(strings.TrimPrefix(alias.Name, runrecord.StageReceiptAliasRoot), "/")
		operationID, err := artifact.ParseID(name)
		receipts[operationID]++
		return err
	}); err != nil {
		t.Fatal(err)
	}
	tools := 0
	for operationID := range receipts {
		if driven[operationID] == "" {
			driven[operationID] = "tool call"
			tools++
		}
	}
	if tools != 2 || toolCalls.Load() != 2 {
		t.Errorf("tool call: two calls ran as %d operations with %d executions, want two operations that each executed", tools, toolCalls.Load())
	}
	t.Logf("tool call: two calls ran as %d operations with %d executions", tools, toolCalls.Load())
	// Each model build left stage receipts under its own operation, which is where a replay looks for them.
	for operationID, entry := range driven {
		if entry == "model build" && receipts[operationID] == 0 {
			t.Errorf("model build: the operation %s left no stage receipt under its id, want the stages of each build under its own operation", operationID)
		}
	}

	// Every alias that names a driven operation lies in a family a reader finds again.
	named := 0
	if err := store.VisitAliases(ctx, "", func(alias overgodb.AliasView) error {
		for operationID, entry := range driven {
			if !strings.Contains(alias.Name, operationID.String()) {
				continue
			}
			named++
			read := false
			for _, root := range operationRecordFamilies {
				read = read || strings.HasPrefix(alias.Name, root)
			}
			if !read {
				t.Errorf("%s: left the alias %s under its operation's id, which no reader of an operation's records finds again: want no such record, or its root in operationRecordFamilies with its reader", entry, alias.Name)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if named == 0 {
		t.Fatalf("no alias names any of the %d driven operations, so the test held nothing: want the stage receipts of the model builds and the tool calls", len(driven))
	}
}

// TestACompletionTheStoreDoesNotHoldFailsItsOperation holds that a workflow's operation completes only with a run the store holds.
//   - a completion that names a run the store lacks ends the operation failed, and the failure names the run
//   - the same workflow completes once its execution commits the run
func TestACompletionTheStoreDoesNotHoldFailsItsOperation(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	workflow := newRecordWorkflow(t, store, "unheld completion")
	handler, err := New(Config{Repository: store, Workflows: workflow}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	run := func() operation.Status {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/generation/run", marshalTestJSON(t, workflow.request("")))
		var submission workflowResponse
		if err := json.Unmarshal(response.Body.Bytes(), &submission); response.Code != http.StatusAccepted || err != nil {
			t.Fatalf("the run answered %d %s (%v), want an operation", response.Code, response.Body.String(), err)
		}
		status, err := handler.operations.Wait(t.Context(), submission.Operation)
		if err != nil {
			t.Fatal(err)
		}
		return status
	}
	workflow.unheld.Store(true)
	if status := run(); status.State != operation.StateFailed || !strings.Contains(status.Failure, workflow.run.String()) {
		t.Fatalf("a completion that names a run the store lacks ended %s with the failure %q, want it failed with the run %s named", status.State, status.Failure, workflow.run)
	}
	workflow.unheld.Store(false)
	if status := run(); status.State != operation.StateCompleted || status.Run == nil || *status.Run != workflow.run {
		t.Fatalf("a completion that names a run the store holds ended %s (%q), want it completed with the run %s", status.State, status.Failure, workflow.run)
	}
}
