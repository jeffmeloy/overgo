package main

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/server"
)

type workflowStub struct {
	executed bool
	run      artifact.ID
}

func (stub *workflowStub) WorkflowCapabilities(context.Context, server.WorkflowKind) ([]server.WorkflowCapability, error) {
	return nil, nil
}

func (stub *workflowStub) ExecuteWorkflow(_ context.Context, _ server.WorkflowKind, _ recipe.Task, _ artifact.ID, _ json.RawMessage, _ operation.Reporter) (operation.Completion, error) {
	stub.executed = true
	return operation.Completion{Run: stub.run}, nil
}

func TestServerRuntimeExecutesDPOWorkflow(t *testing.T) {
	run, err := artifact.IdentifyBytes(artifact.KindRun, []byte("run"))
	if err != nil {
		t.Fatal(err)
	}
	stub := &workflowStub{run: run}
	runtime := &serverRuntime{WorkflowWorkspaceAPI: stub}
	var workspace server.WorkflowWorkspaceAPI = runtime
	if _, err := workspace.ExecuteWorkflow(context.Background(), server.WorkflowTraining, recipe.TaskTraining, artifact.ID{}, nil, nil); err != nil || !stub.executed {
		t.Fatalf("executed=%v err=%v", stub.executed, err)
	}
}
